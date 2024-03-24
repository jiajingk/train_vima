from typing import List, Tuple
from playground.typing import (
    NormalizedTraj,
    Device,
    VIMAPolicy,
    Tensor,
    DecodeMeta,
    Action,
    ForwardMetaData,
    PredDist,
    Criterion,
    ActionAxisWeight,
    TaskWeight
)
from playground.util.prepare_batch import prepare_batch_obs_action_tokens
from playground.util.prompt import get_task_class
from playground.util.reduce import reduce_weighted_step_total_loss
from playground.util.measure import (
    measure_traj_individual_loss, 
    to_flatten_step_measure
)
from datetime import datetime
import uuid


def freeze_t5_except_last_2_layer(policy: VIMAPolicy) -> VIMAPolicy:
    """
    layer in paper refers to the transfomer layer as suggest here
    https://github.com/vimalabs/VIMA/issues/16#issuecomment-1622973970
    """
    for param in policy.t5_prompt_encoder.parameters():
        param.requires_grad = False
    last_two_layers = (
        policy.t5_prompt_encoder.t5.encoder.block[-1],
        policy.t5_prompt_encoder.t5.encoder.block[-2],
    )
    for layer in last_two_layers:
        for param in layer.parameters():
            param.requires_grad = True
    return policy


def decode_observation_and_prompt( 
        trajs: List[NormalizedTraj], 
        device: Device, 
        policy: VIMAPolicy 
    ) -> Tuple[Tensor, List[DecodeMeta]]: 
    traj_time_step_length = len(trajs) 
    data_batch = prepare_batch_obs_action_tokens(trajs, device, policy) 
    tokens_out = policy.xattn_gpt( 
        obs_action_tokens=data_batch["obs_action_tokens"].transpose(1, 0), 
        obs_action_masks=data_batch["obs_action_masks"], 
        obs_action_position_ids=data_batch["position_ids"], 
        prompt_tokens=data_batch["prompt_tokens"].transpose(1, 0), 
        prompt_mask=data_batch["prompt_masks"], 
        prompt_position_ids=data_batch["prompt_position_ids"], 
    ) 
    decode_metas: List[DecodeMeta] = [ 
        { 
            "n_max_objs": data_batch["n_max_objs"][t_step], 
            "is_rotation": data_batch["is_rotation"][t_step], 
            "target_actions": data_batch["target_actions"][t_step], 
            "token_lengths": data_batch["token_lengths"][t_step], 
            "prompt": data_batch["prompt"][t_step],
            "task": data_batch["task"][t_step],
        } for t_step in range(traj_time_step_length) 
    ] 
    return tokens_out, decode_metas 


def extract_action(
        tokens: Tensor, 
        batch_id: int, 
        content_lenght: int,
        num_obj_detected: int
    ) -> Tensor:
    return tokens[:content_lenght][num_obj_detected - 1 :: num_obj_detected + 1, batch_id, :].clone()


def forward(
        trajs: List[NormalizedTraj], 
        device: Device, 
        policy: VIMAPolicy
    ) -> List[Tuple[PredDist, Action, ForwardMetaData]]:
    tokens_out, decode_metas = decode_observation_and_prompt(
        trajs, device, policy
    )
    results = []
    for i, decode_meta in enumerate(decode_metas):
        n_max_objs = decode_meta["n_max_objs"]
        predicted_action_tokens = extract_action(
            tokens_out, i, decode_meta["token_lengths"], n_max_objs
        )
        dist_dict = policy.forward_action_decoder(predicted_action_tokens)
        target_action = decode_meta["target_actions"]
        results.append(
            (
                dist_dict, 
                target_action, 
                {
                    "is_rotation": decode_meta["is_rotation"],
                    "action_length": target_action["pose0_position"].shape[0],
                    "prompt": decode_meta["prompt"],
                    "task": decode_meta["task"],
                }
            )
        )
    return results



def measure_sample_loss(
        sample_pred: Tuple[PredDist, Action, ForwardMetaData],
        criterion: Criterion,
        axis_weights: ActionAxisWeight,
        task_weights: TaskWeight,
        batch_id: int,
        epoch_id: int,
    ) -> Tensor:
    pred_dist, target_action, forward_meta = sample_pred
    traj_imitation_loss = measure_traj_individual_loss(
        pred_dist, target_action, forward_meta, criterion
    )
    step_losses = [
        reduce_weighted_step_total_loss(
            step_loss, 
            forward_meta,
            axis_weights,
            task_weights
        ) 
            for step_loss in traj_imitation_loss
    ]
    sample_loss_log = [
        { 
            **to_flatten_step_measure(step_loss),
            "epoch": epoch_id,
            "batch": batch_id,
            "step_length": len(step_losses),
            "step_id": i,
            "prompt": forward_meta['prompt'],
            "task": get_task_class(forward_meta['prompt'])
        }
            for i, step_loss in enumerate(traj_imitation_loss)
    ]
    sample_loss = 0
    for step_loss in step_losses:
        sample_loss += step_loss
    return sample_loss / len(step_losses), sample_loss_log


def generate_run_id() -> str:
    today: str = datetime.today().strftime('%Y-%m-%d-%H-%M')
    return f"{today}_{uuid.uuid4().hex[:8]}"