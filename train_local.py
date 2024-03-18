from playground.util.replay_env import (
    normalize_traj, 
    load_normalized_traj
)
from playground.util.policy import get_policy_and_cfg
from playground.util.train import ( 
    measure_traj_individual_loss,
    reduce_weighted_step_total_loss,
    forward,
    freeze_t5_except_last_2_layer,
)
from playground.util.loss_scaling import (
    get_action_weigts,
    get_default_task_weight,
    calculate_action_weight_factor,
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
from playground.typing import (
    NormalizedTraj,
    Device,
    VIMAPolicy,
    PredDist, 
    Action, 
    ForwardMetaData,
    Tensor,
    Criterion,
    DataFrameTrainLogSchema,
    TrainParam,
    DDPParam,
    OptimizerParam,
    DatasetParam,
    CosAnnealingParam,
    TrainHistory
)
from playground.util.prompt import get_task_class
from torch.utils.data import DataLoader
from tqdm import tqdm
from vima.policy import VIMAPolicy
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime
from glob import glob
import torch
import os
import pandas as pd
import math
import uuid


DDP_PARAM: Optional[DDPParam] = None

def get_lr_param() -> CosAnnealingParam:
    return {
        "warmup_end_at_iters": 7000,
        "flatten_end_at_iters": 24000,
        "lr_decay_end_at_iters": 48000,
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
        "data_pct_usage": 0.1,
        "total_data_size_per_task": 256,
        "validation_pct": 0.1,
        "source": "local",
        "tasks": [
            'visual_manipulation',
            'manipulate_old_neighbor',
            'novel_adj',
            'novel_noun',
            'pick_in_order_then_restore',
            'rearrange',
            'rearrange_then_restore',
            'same_profile',
            'rotate',
            'scene_understanding',
            'simple_manipulation',
            'sweep_without_exceeding',
            'follow_order',
            'twist'
        ]
    }


def get_train_param() -> TrainParam:
    return {
        "total_epoch": 300,
        "local_batch_size": 32,
        "distributed": False,
        "model_size": '2M'
    }


def flatten_dict(d: Dict[str, Union[str, int, float]], parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_clip_grad_norm() -> float:
    return get_optimizer_param()["clip_norm"]


def get_local_batch_size() -> int:
    return get_train_param()["local_batch_size"]


def get_ddp_param() -> DDPParam:
    if DDP_PARAM is None:
        raise ValueError("unable to retrieve ddp param")
    return {
        "local_rank": int(DDP_PARAM.local_rank),
        "master_ip": str(DDP_PARAM.master_ip),
        "master_port": str(DDP_PARAM.master_port),
        "world_size": int(DDP_PARAM.world_size)
    }


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


def measure_lr(
        dataset_param: DatasetParam,
        train_param: DatasetParam,
        batch_id: int, 
        epoch_id: int
    ):
    epoch_size = (
        int(dataset_param["total_data_size_per_task"] * len(dataset_param["tasks"]) 
            * dataset_param["data_pct_usage"])
    )
    batch_size = (
        train_param["local_batch_size"] 
            if train_param["distributed"] is False 
            else train_param["local_batch_size"] * get_ddp_param()["world_size"]
    )
    batch_count_per_epoch = epoch_size // batch_size
    if epoch_size % batch_size != 0:
        batch_count_per_epoch += 1
    current_total_batch_count = batch_id + epoch_id * batch_count_per_epoch
    return get_lr(current_total_batch_count)


def update_and_get_lr(
        optimizer: torch.optim.AdamW, 
        batch_id: int, 
        epoch_id: int
    ) -> float:
    lr = measure_lr(
        get_dataset_param(),
        get_train_param(),
        batch_id,
        epoch_id
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def measure_unweighted_loss(
        logs: DataFrameTrainLogSchema, 
        epoch_id: int,
    ) -> float:
    unweighted_sample_cols = [
        col for col in logs.columns 
            if 'unweigted_sample_loss' in col
    ]
    desired_cols = logs.columns.isin(unweighted_sample_cols)
    epoch_id_col_candidates = [
        col for col in logs.columns 
            if 'epoch_id' in col
    ]
    if len(epoch_id_col_candidates) == 0:
        raise ValueError(f"could not find epoch_id column in {logs.columns }")
    epoch_id_col = epoch_id_col_candidates[0]
    epoch_loss = ( 
        logs
        .loc[logs[epoch_id_col] == epoch_id]
        .iloc[:, desired_cols]
        .sum(axis=1)
        .mean()
    )
    return float(epoch_loss)


class VimaPolicyWraper(torch.nn.Module):
    def __init__(
        self,
        *,
        single_process_policy: VIMAPolicy,
        device: Device
        ):
        super().__init__()
        self.policy = single_process_policy
        self.device = device


    def forward(
            self, 
            data: List[NormalizedTraj]
        ) -> List[Tuple[PredDist, Action, ForwardMetaData]]:
        batch_preds = forward(
            data,
            self.device,
            self.policy
        )
        return batch_preds

BatchLoss = Tensor
LogRecord = Dict[str, float]

def batch_forward(
        policy: VimaPolicyWraper, 
        data: List[NormalizedTraj],
        criterion: Criterion,
    ) -> Tuple[BatchLoss, List[LogRecord]]:
    batch_forward = policy(data)
    batch_losses = [
        measure_traj_individual_loss(
            pred_dist, target_action, forward_meta, criterion
        ) for pred_dist, target_action, forward_meta in batch_forward
    ]
    axis_weight = get_action_weigts(
        batch_losses,
        "default"
    )
    scaling_factor = calculate_action_weight_factor(axis_weight)
    task_weight = get_default_task_weight(1.0)
    unweigted_sample_losses = [
        reduce_traj_loss_in_time_axis(traj_loss, lambda _: 1.0)
            for traj_loss in batch_losses
    ]
    weighted_sample_losses = [
        reduce_weighted_step_total_loss(
            unweigted_sample_loss, 
            forward_meta,
            axis_weight,
            task_weight,
        ) * scaling_factor
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
            "task": get_task_class(forward_meta['prompt']),
            "local_rank": 0 if DDP_PARAM is None else get_ddp_param()["local_rank"]
        }
            for unweigted_sample_loss, (_, _, forward_meta) in zip(unweigted_sample_losses, batch_forward)
    ]
    batch_loss = torch.sum(
        torch.stack(weighted_sample_losses)
    ) / len(weighted_sample_losses)
    return batch_loss, batch_loss_log


def validate(
        ddp_policy: VimaPolicyWraper,
        dataloader: DataLoader, 
        criterion: torch.nn.CrossEntropyLoss,
        epoch_id: int,
        curr_lr: float
    ) -> Tuple[float, List[LogRecord]]:
    ddp_policy.eval()
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
        epoch_logs += batch_logs
        epoch_loss += batch_loss.item()
        pbar.set_description(f"valid {epoch_id}: {batch_loss.item()}")
    return epoch_loss / (batch_id + 1), [
        flatten_dict(
            {"validation": log} 
        ) for log in epoch_logs
    ]


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
        dataloader: DataLoader, 
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.AdamW,
        epoch_id: int,
    ) -> Tuple[float, List[LogRecord]]:
    policy.train()
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
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            get_clip_grad_norm()
        )
        optimizer.step()
        epoch_logs += batch_logs
        epoch_loss += batch_loss.item()        
        pbar.set_description(f"train {epoch_id}: {batch_loss.item()}")
    epoch_loss /= (batch_id + 1)
    return epoch_loss, epoch_logs


def get_parent_model_path(model_repo_folder: str) -> Tuple[str, bool]:
    model_weight_locations = list(glob(os.path.join(model_repo_folder, '*.ckpt')))
    if len(model_weight_locations) == 0:
        model_name = get_train_param()["model_size"]
        return os.path.join('.', f'{model_name}.ckpt'), True
    else:
        model_name = model_weight_locations[0].split('/')[-1].split('.')[0]
        return os.path.join(model_repo_folder, f'{model_name}.ckpt'), False


def main(model_repo_folder):
    run_id: str = uuid.uuid4().hex
    today: str = datetime.today().strftime('%Y-%m-%d')
    train_log_csv_name = f'train_{run_id}_{today}_log.csv'
    valid_log_csv_name = f'valid_{run_id}_{today}_log.csv'
    train_loader, valid_loader = get_dataloader(
        get_train_param(),
        get_dataset_param(),
    )
    model_path, from_scratch= get_parent_model_path(model_repo_folder)
    policy, cfg, train_history = get_policy_and_cfg(model_path, 'cuda', from_scratch=from_scratch)
    freeze_t5_except_last_2_layer(policy)
    epochs = get_train_param()["total_epoch"]
    if from_scratch is True:
        inital_epoch = 0
    else:
        inital_epoch = train_history["last_epoch"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=get_optimizer_param()["inital_lr"], 
        weight_decay=get_optimizer_param()["weight_decay"]
    )
    assert optimizer.__class__.__name__ == get_optimizer_param()["optimizer_name"]
    
    ddp_policy = VimaPolicyWraper(single_process_policy=policy, device='cuda')
    for epoch in range(inital_epoch + 1, epochs):
        weighted_epoch_loss, epoch_logs = train_one_epoch(
            ddp_policy,
            train_loader,
            criterion,
            optimizer,
            epoch,
        )
        if epoch % 50 == 0 and epoch > 0:
            save_file_name = f'{run_id}_{epoch}_{today}.ckpt'
            path = os.path.join('saved_model', save_file_name)
            train_history: TrainHistory = {
                "last_epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict()
            }
            ckpt = {
                'cfg': cfg,
                'state_dict': policy.state_dict(),
                'history': train_history
            }
            torch.save(ckpt, path)
        logs = pd.DataFrame(data=epoch_logs)
        if epoch == 0:
            logs.to_csv(train_log_csv_name, index=False)
        else:
            logs.to_csv(train_log_csv_name, mode='a', header=False, index=False)
        print("train", measure_unweighted_loss(logs, epoch))
        weighted_epoch_loss, valid_logs = validate(
            ddp_policy,
            valid_loader, 
            criterion,
            epoch,
            get_current_lr(optimizer)
        )
        logs = pd.DataFrame(data=valid_logs)
        if epoch == 0:
            logs.to_csv(valid_log_csv_name, index=False)
        else:
            logs.to_csv(valid_log_csv_name, mode='a', header=False, index=False)
        print("valid", measure_unweighted_loss(logs, epoch))
    save_file_name = f'{run_id}_{epoch}_{today}.ckpt'
    path = os.path.join('saved_model', save_file_name)
    ckpt = {
        'cfg': cfg,
        'state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(ckpt, path)
    

if __name__ == "__main__":
    main(os.path.join('.', 'parent_model', '*.ckpt'))