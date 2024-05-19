from playground.util.replay_env import (
    normalize_traj, 
    load_normalized_traj
)
from playground.util.policy import get_policy_and_cfg, VimaPolicyWraper
from playground.util.train import ( 
    measure_traj_individual_loss,
    reduce_weighted_step_total_loss,
    freeze_t5_except_last_2_layer,
)
from playground.util.loss_scaling import (
    get_action_weigts,
    get_task_weights,
    get_default_axis_weight
)
from playground.util.reduce import (
    reduce_traj_loss_in_time_axis
)
from playground.dataloader import (
    get_dataloader
)
from playground.source.s3 import (
    load_batch, 
)
from playground.source.tf_record import (
    deseralize,
)
from playground.util.log import (
    measure_unweighted_loss_per_attribute,
    measure_unweighted_loss_per_task,
    measure_avg_unweighted_loss,
    measure_avg_lr,
    flatten_dict
)
from playground.ddp import init_process
from playground.typing import (
    NormalizedTraj,
    Tensor,
    Criterion,
    TrainParam,
    DDPParam,
    OptimizerParam,
    DatasetParam,
    CosAnnealingParam,
    TrainHistory,
    DistributedDataLoader,
    TimedLog,
    PredDist, 
    Action, 
    ForwardMetaData,
    InitalizeMode,
    ActionAxisWeight,
    FlattenedStepMeasure
)
from playground.util.prompt import get_task_class
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional, Union, Callable
from datetime import datetime
from glob import glob
from collections import namedtuple
import torch
import os
import pandas as pd
import math
import argparse
import wandb


BatchLoss = Tensor
LogRecord = Dict[str, float]

DDP_PARAM: Optional[DDPParam] = None
import math 

def get_wandb_param():
    return {
        "project": "vima",
        "group": "ddp",
        "job_type": "train"
    }

def get_lr_param() -> CosAnnealingParam:
    return {
        "warmup_end_at_iters": 7000,
        "flatten_end_at_iters": 240000,
        "lr_decay_end_at_iters": 960000,
        "learning_rate": 1e-4,
        "min_lr": 1e-7, 
    }


def get_optimizer_param() -> OptimizerParam:
    return {
        "clip_norm": 1.0,
        "inital_lr": get_lr_param()["learning_rate"],
        "optimizer_name": "AdamW",
        "weight_decay": 0.0
    }

def get_dataset_param() -> DatasetParam:
    return  {
        "data_pct_usage": 1.0,
        "total_data_size_per_task": 40000,
        "validation_pct": 0.00,
        "source": "s3://vima",
        "tasks": [
            "follow_order",
            "manipulate_old_neighbor",
            "novel_adj",
            "novel_noun",
            "pick_in_order_then_restore",
            "rearrange_then_restore",
            "rearrange",
            "rotate",
            "same_profile",
            "scene_understanding",
            "simple_manipulation",
            "sweep_without_exceeding",
            "twist",
        ]
    }


def get_train_param() -> TrainParam:
    return {
        "model_size": "2M",
        "total_epoch": 50,
        "local_batch_size": 16,
        "distributed": True,
    }


def get_ddp_param() -> DDPParam:
    if DDP_PARAM is None:
        return {
            "local_rank": 0,
            "master_ip": "",
            "master_port": "",
            "world_size": 1,
            "backend": "",
            "socket": ""
        }
    return {
        "local_rank": int(DDP_PARAM.local_rank),
        "master_ip": str(DDP_PARAM.master_ip),
        "master_port": str(DDP_PARAM.master_port),
        "world_size": int(DDP_PARAM.world_size),
        "backend": "nccl",
        "socket": "ens5"
    }


def get_clip_grad_norm() -> float:
    return get_optimizer_param()["clip_norm"]


def get_local_batch_size() -> int:
    return get_train_param()["local_batch_size"]


def get_current_lr(optimizer: torch.optim.AdamW) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
    raise ValueError("couldn't find any lr in all param_groups")


def get_lr(it: int) -> float:
    lr_param = get_lr_param()
    warmup_iters = lr_param["warmup_end_at_iters"]
    flatten_iters = lr_param["flatten_end_at_iters"]
    learning_rate = lr_param["learning_rate"]
    lr_decay_iters = lr_param["lr_decay_end_at_iters"]
    min_lr = lr_param["min_lr"]
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if warmup_iters <= it < flatten_iters:
        return learning_rate
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - flatten_iters) / (lr_decay_iters - flatten_iters)
    assert 0 <= decay_ratio <= 1, f"{decay_ratio = }, {it = }"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch_per_epoch(
        dataset_param: DatasetParam,
        train_param: TrainParam,
        ddp_param: DDPParam,
        is_train: bool = True
    ):
    if is_train:
        scaling = 1.0
    else:
        scaling = dataset_param["validation_pct"]
    epoch_size = ( 
        int(
            dataset_param["total_data_size_per_task"] 
            * scaling 
            * dataset_param["data_pct_usage"]
        ) 
        * len(dataset_param["tasks"]) 
    )
    batch_size = (
        train_param["local_batch_size"] 
            if train_param["distributed"] is False 
            else train_param["local_batch_size"] * ddp_param["world_size"]
    )
    if epoch_size % batch_size != 0:
        return epoch_size // batch_size + 1
    return epoch_size // batch_size


def get_total_batch_count(
        dataset_param: DatasetParam,
        train_param: TrainParam,
        ddp_param: DDPParam,
        batch_id: int, 
        epoch_id: int,
        is_train: bool = True
    ) -> int:
    batch_count_per_epoch = get_batch_per_epoch(dataset_param, train_param, ddp_param, is_train)
    current_total_batch_count = batch_id + epoch_id * batch_count_per_epoch
    return current_total_batch_count

def get_train_batch_count(
        train_batch_id: int, 
        epoch_id: int,
    ) -> int:
    return get_total_batch_count(
        get_dataset_param(),
        get_train_param(),
        get_ddp_param(),
        train_batch_id, epoch_id, is_train=True
    )

def get_valid_batch_count(
        valid_batch_id: int, 
        epoch_id: int,
    ) -> int:
    return get_total_batch_count(
        get_dataset_param(),
        get_train_param(),
        get_ddp_param(),
        valid_batch_id, epoch_id, is_train=False
    )


def measure_lr(
        dataset_param: DatasetParam,
        train_param: TrainParam,
        ddp_param: DDPParam,
        batch_id: int, 
        epoch_id: int
    ):
    current_total_batch_count = get_total_batch_count(
        dataset_param, 
        train_param, 
        ddp_param,
        batch_id, 
        epoch_id, 
        is_train=True
    )
    return get_lr(current_total_batch_count)


def update_and_get_lr(
        optimizer: torch.optim.AdamW, 
        batch_id: int, 
        epoch_id: int
    ) -> float:
    lr = measure_lr(
        get_dataset_param(),
        get_train_param(),
        get_ddp_param(),
        batch_id,
        epoch_id
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

MeasureMethod = Callable[[List[LogRecord], Tuple[str, int], str], TimedLog]
def log_to_wandb(
        measure_methods: List[MeasureMethod],
        logs: List[LogRecord], 
        timestamp: Tuple[str, int],
        prefix: str = ''):
    wandb_logs: List[TimedLog] = [
        measure_method(
            logs, timestamp, prefix
        ) for measure_method in measure_methods
    ]
    for wandb_log in wandb_logs:
        wandb.log(flatten_dict({
            "measure": wandb_log["measure"],
            "timestamp": {wandb_log["timestamp"][0]: wandb_log["timestamp"][1]}
        }))


def get_custom_scaling() -> ActionAxisWeight:
    weight = get_default_axis_weight(1.0)
    weight["pose0_position_0"] = 1 / math.log(50)
    weight["pose0_position_1"] = 1 / math.log(100)
    weight["pose1_position_0"] = 1 / math.log(50)
    weight["pose1_position_1"] = 1 / math.log(100)
    weight["pose0_rotation_0"] = 1 / math.log(50)
    weight["pose0_rotation_1"] = 1 / math.log(50)
    weight["pose0_rotation_2"] = 1 / math.log(50)
    weight["pose0_rotation_3"] = 1 / math.log(50)
    weight["pose1_rotation_0"] = 1 / math.log(50)
    weight["pose1_rotation_1"] = 1 / math.log(50)
    weight["pose1_rotation_2"] = 1 / math.log(50)
    weight["pose1_rotation_3"] = 1 / math.log(50)
    return weight


def apply_rotation_mask(
        traj_losses: FlattenedStepMeasure, 
        forward_meta: ForwardMetaData
    ) -> FlattenedStepMeasure:
    if forward_meta["is_rotation"]:
        return traj_losses
    for attr in traj_losses:
        if 'rotation' in attr:
            traj_losses[attr] *= 0
    return traj_losses


def batch_forward(
        policy: VimaPolicyWraper, 
        data: List[NormalizedTraj],
        criterion: Criterion,
    ) -> Tuple[BatchLoss, List[LogRecord]]:
    batch_forward: List[Tuple[PredDist, Action, ForwardMetaData]] = policy(data)
    batch_losses = [
        measure_traj_individual_loss(
            pred_dist, target_action, forward_meta, criterion, rotation_mask=True
        ) for pred_dist, target_action, forward_meta in batch_forward
    ]
    axis_weight = get_custom_scaling()
    task_weight = get_task_weights(
        None,
        None,
        "default"
    )
    unweigted_sample_losses = [
        reduce_traj_loss_in_time_axis(traj_loss, lambda _: 1.0)
            for traj_loss in batch_losses
    ]
    unweigted_sample_losses = [
        apply_rotation_mask(traj_loss, forward_meta)
            for (traj_loss, (_, _, forward_meta)) in zip(unweigted_sample_losses, batch_forward)
    ]
    weighted_sample_losses = [
        reduce_weighted_step_total_loss(
            unweigted_sample_loss, 
            forward_meta,
            axis_weight,
            task_weight,
        )
            for unweigted_sample_loss, (_, _, forward_meta) in zip(unweigted_sample_losses, batch_forward)
    ]
    batch_loss_log = [
        { 
            **flatten_dict(
                {
                    "unweigted_sample_loss": {k: v.item() for k, v in unweigted_sample_loss.items()},
                    "axis_weight": axis_weight,
                    "task_weight": task_weight
                }
            ),
            "step_size": forward_meta["action_length"],
            "task": forward_meta["task"],
            "local_rank": 0 if DDP_PARAM is None else get_ddp_param()["local_rank"]
        }
            for unweigted_sample_loss, (_, _, forward_meta) in zip(unweigted_sample_losses, batch_forward)
    ]
    batch_loss = torch.sum(torch.stack(weighted_sample_losses)) / len(weighted_sample_losses)
    return batch_loss, batch_loss_log


@torch.no_grad()
def validate(
        ddp_policy: VimaPolicyWraper,
        dataloader: Union[DistributedDataLoader, DataLoader], 
        criterion: torch.nn.CrossEntropyLoss,
        epoch_id: int,
        curr_lr: float
    ) -> Tuple[float, List[LogRecord]]:
    ddp_policy.eval()
    if get_train_param()["distributed"] is True:
        dataloader.sampler.set_epoch(epoch_id)
    epoch_logs: List[LogRecord] = []
    epoch_loss = 0
    for batch_id, records in (pbar := tqdm(enumerate(dataloader))):
        trajs = records_to_trajs(records)   
        batch_loss, batch_logs = batch_forward(
            ddp_policy,
            trajs,
            criterion
        )
        batch_logs = [
            {
                **log_record,
                "batch_id": batch_id,
                "epoch_id": epoch_id,
                "lr": curr_lr
            } for log_record in batch_logs
        ]
        write_log_to_csv(
            batch_logs, wandb.run.id, f'validate_{epoch_id}'
        )
        log_to_wandb(
            [
                measure_unweighted_loss_per_task,
                measure_unweighted_loss_per_attribute,
                measure_avg_unweighted_loss,
            ],
            batch_logs, ("valid_batch", get_valid_batch_count(batch_id, epoch_id)), 'valid__batch__'
        )
        epoch_logs += batch_logs
        epoch_loss += batch_loss.item()
        pbar.set_description(f"valid {epoch_id}: weighted loss {batch_loss.item()}")
    return epoch_loss / (batch_id + 1), epoch_logs


def records_to_trajs(
        records: List[Union[str, Tuple[str, str]]]
    ) -> List[NormalizedTraj]:
    if get_dataset_param()["source"] == "local":
        task_ids, task_names = records
        return [
            load_normalized_traj(int(task_id), task_name) 
                for (task_id, task_name) in zip(task_ids, task_names)
        ]
    if get_dataset_param()["source"] == "s3://vima":
        raw_records = load_batch(records, len(records))
        return list(normalize_traj(deseralize(raw_record)) for raw_record in raw_records)
    raise AssertionError(f'unknown source {get_dataset_param()["source"]}')


def train_one_epoch(
        policy: VimaPolicyWraper,
        dataloader: Union[DistributedDataLoader, DataLoader], 
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.AdamW,
        epoch_id: int,
    ) -> Tuple[float, List[LogRecord]]:
    policy.train()
    if get_train_param()["distributed"] is True:
        dataloader.sampler.set_epoch(epoch_id)
    epoch_logs: List[LogRecord] = []
    epoch_loss = 0
    for batch_id, records in (pbar := tqdm(enumerate(dataloader))):
        trajs = records_to_trajs(records)
        curr_lr = update_and_get_lr(optimizer, batch_id, epoch_id)
        optimizer.zero_grad()
        batch_loss, batch_logs = batch_forward(
            policy,
            trajs,
            criterion
        )
        batch_logs = [
            {
                **log_record,
                "batch_id": batch_id,
                "epoch_id": epoch_id,
                "lr": curr_lr
            } for log_record in batch_logs
        ]
        write_log_to_csv(
            batch_logs, wandb.run.id, f'train_{epoch_id}'
        )
        log_to_wandb(
            [
                measure_unweighted_loss_per_task,
                measure_unweighted_loss_per_attribute,
                measure_avg_lr,
                measure_avg_unweighted_loss,
            ],
            batch_logs, ("train_batch", get_train_batch_count(batch_id, epoch_id)), 'train__batch__'
        )
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            get_clip_grad_norm()
        )
        optimizer.step()
        epoch_logs += batch_logs
        epoch_loss += batch_loss.item()        
        pbar.set_description(f"train {epoch_id}: weighted loss {batch_loss.item()}")
    epoch_loss /= (batch_id + 1)
    return epoch_loss, epoch_logs


def get_parent_model_path(model_repo_folder: str) -> Tuple[str, bool]:
    model_weight_locations = list(glob(os.path.join(model_repo_folder, '*.ckpt')))
    if len(model_weight_locations) == 0:
        model_name = get_train_param()["model_size"]
        return os.path.join('.', f'{model_name}.ckpt'), True
    else:
        return model_weight_locations[0], False


def write_log_to_csv(logs: List[LogRecord], run_id: str, log_type: str):
    log_file_path = f"{log_type}_{run_id}.csv"
    log_df = pd.DataFrame(data=logs)
    if not os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, index=False)
    else:
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)

def write_model_checkpoint(
        run_id: str,
        epoch: int,
        optimizer_state_dict: Dict,
        policy_state_dict: Dict,
        cfg: Dict
    ):
    if (get_train_param()["distributed"] is True 
            and get_ddp_param()["local_rank"] != 0):
        return
    save_file_name = f'{run_id}_{epoch}.ckpt'
    path = os.path.join('..', 'saved_model', save_file_name)
    train_history: TrainHistory = {
        "last_epoch": epoch,
        "optimizer_state_dict": optimizer_state_dict
    }
    ckpt = {
        'cfg': cfg,
        'state_dict': policy_state_dict,
        'history': train_history
    }
    torch.save(ckpt, path)


@torch.no_grad()
def eval_ddp(
        model_repo_folder: str,
        initalize_mode: InitalizeMode
    ):
    model_path, from_scratch = get_parent_model_path(model_repo_folder)
    if '2M' in model_path:
        prefix = ''
    else:
        prefix = 'module.'
    if from_scratch is False:
        mode = initalize_mode
    else:
        mode: InitalizeMode = 'random_init'
    policy, _, _ = get_policy_and_cfg(
        model_path, 
        'cuda', 
        prefix=prefix, 
        mode=mode
    )
    freeze_t5_except_last_2_layer(policy)
    policy = VimaPolicyWraper(single_process_policy=policy, device='cuda')
    if get_train_param()["distributed"] is True:
        init_process(
            get_ddp_param()
        )
        policy = DDP(
            policy,
            find_unused_parameters=True,
            static_graph=True,
        )
    dataloader, _ = get_dataloader(
        get_train_param(),
        get_dataset_param(),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=get_optimizer_param()["inital_lr"], 
        weight_decay=get_optimizer_param()["weight_decay"]
    )
    assert optimizer.__class__.__name__ == get_optimizer_param()["optimizer_name"]

    model_parent = (
        f"{get_train_param()['model_size']} from scratch" 
            if from_scratch is True else os.path.basename(model_path)
    )
    wandb_config = {
        "ddp_param": get_ddp_param(),
        "train_param": get_train_param(),
        "dataset_param": get_dataset_param(),
        "lr_param": get_lr_param(),
        "parent_model": model_parent
    }
    wandb.init(
        project=get_wandb_param()["project"],
        config=wandb_config,
        group=get_wandb_param()["group"],
        job_type=get_wandb_param()["job_type"]
    )
    _, valid_logs = validate(
        policy,
        dataloader, 
        criterion,
        0,
        get_current_lr(optimizer)
    )
    log_to_wandb(
        [
            measure_unweighted_loss_per_task,
            measure_unweighted_loss_per_attribute,
            measure_avg_unweighted_loss,
        ],
        valid_logs, ("valid_epoch", 0), 'valid__epoch__'
    )

def main_ddp(
        model_repo_folder: str,
        initalize_mode: InitalizeMode
    ):
    model_path, from_scratch = get_parent_model_path(model_repo_folder)
    if '2M' in model_path:
        prefix = ''
    else:
        prefix = 'module.'
    if from_scratch is False:
        mode = initalize_mode
    else:
        mode: InitalizeMode = 'random_init'
    policy, cfg, train_history = get_policy_and_cfg(
        model_path, 
        'cuda', 
        prefix=prefix, 
        mode=mode
    )
    freeze_t5_except_last_2_layer(policy)
    policy = VimaPolicyWraper(single_process_policy=policy, device='cuda')
    if get_train_param()["distributed"] is True:
        init_process(
            get_ddp_param()
        )
        policy = DDP(
            policy,
            find_unused_parameters=True,
            static_graph=True,
        )
    train_loader, valid_loader = get_dataloader(
        get_train_param(),
        get_dataset_param(),
    )
    epochs = get_train_param()["total_epoch"]
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=get_optimizer_param()["inital_lr"], 
        weight_decay=get_optimizer_param()["weight_decay"]
    )
    assert optimizer.__class__.__name__ == get_optimizer_param()["optimizer_name"]
    if train_history is None:
        inital_epoch = 0
    else:
        inital_epoch = train_history["last_epoch"] + 1
        optimizer.load_state_dict(train_history["optimizer_state_dict"])
    model_parent = (
        f"{get_train_param()['model_size']} from scratch" 
            if from_scratch is True else os.path.basename(model_path)
    )
    wandb_config = {
        "ddp_param": get_ddp_param(),
        "train_param": get_train_param(),
        "dataset_param": get_dataset_param(),
        "lr_param": get_lr_param(),
        "parent_model": model_parent
    }
    wandb.init(
        project=get_wandb_param()["project"],
        config=wandb_config,
        group=get_wandb_param()["group"],
        job_type=get_wandb_param()["job_type"]
    )
    today: str = datetime.today().strftime('%Y-%m-%d')
    run_id: str = f"{today}_{wandb.run.name}"
    for epoch in range(inital_epoch, epochs):
        _, train_logs = train_one_epoch(
            policy,
            train_loader,
            criterion,
            optimizer,
            epoch,
        )
        log_to_wandb(
            [
                measure_unweighted_loss_per_task,
                measure_unweighted_loss_per_attribute,
                measure_avg_lr,
                measure_avg_unweighted_loss,
            ],
            train_logs, ("train_epoch", epoch), 'train__epoch__'
        )
        
        write_model_checkpoint(
            run_id, 
            epoch, 
            optimizer.state_dict(),
            policy.state_dict(),
            cfg
        )
        if valid_loader is not None:
            _, valid_logs = validate(
                policy,
                valid_loader, 
                criterion,
                epoch,
                get_current_lr(optimizer)
            )
            log_to_wandb(
                [
                    measure_unweighted_loss_per_task,
                    measure_unweighted_loss_per_attribute,
                    measure_avg_unweighted_loss,
                ],
                valid_logs, ("valid_epoch", epoch), 'valid__epoch__'
            )
    write_model_checkpoint(
        run_id, 
        epoch, 
        optimizer.state_dict(),
        policy.state_dict(),
        cfg
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--master_ip", type=str,  default='localhost')
    parser.add_argument("--master_port", type=str, default='29500')
    parser.add_argument("--train_mode", type=str, default='random_init')
    args = parser.parse_args()
    DDP_PARAM = namedtuple('DDPP', 'local_rank world_size master_ip master_port')(
        local_rank=args.local_rank,
        world_size=args.world_size,
        master_ip=args.master_ip,
        master_port=args.master_port
    )
    assert get_ddp_param()["world_size"] * get_train_param()["local_batch_size"] == 128
    assert args.train_mode in ('random_init', 'ckpt_init', 'continous_from_ckpt',)
    main_ddp(os.path.join('..', 'parent_model'), args.train_mode)