from typing import List, Dict
from vima.utils import any_concat
from torch import Tensor

from playground.typing import (
    Device, NormalizedTraj, VIMAPolicy, DataBatch
)
from playground.util.prepare_single import (
    prepare_single_record_obs, 
    prepare_single_record_prompt,
    prepare_single_record_optional_action,
    prepare_single_obs_action_tokens
)
import torch



def prepare_batch_obs_action_tokens(
        trajs: List[NormalizedTraj], 
        device: Device, 
        policy: VIMAPolicy
    ) -> DataBatch:
    data_batch: Dict[str, List] = {
        "obs_action_tokens": [],
        "obs_action_masks": [],
        "prompt_tokens": [],
        "prompt_masks": [],
        "position_ids": [],
        "prompt_position_ids": [],
        "n_max_objs": [],
        "target_actions": [],
        "token_lengths": [],
        "is_rotation": [],
        "prompt": [],
        "task": [],
    }
    for traj in trajs:
        is_rotation = bool( 
            any(
                element in traj['meta']['prompt'].lower() for element in ( 'rotate', 'twist', ) 
            ) 
        )
        obs_tokens_to_forward, obs_masks_to_forward = prepare_single_record_obs(traj, device, policy)
        prompt_tokens, prompt_masks = prepare_single_record_prompt(traj, device, policy)
        action_in, target_action = prepare_single_record_optional_action(traj, device, policy)
        prompt_position_ids = torch.cumsum(prompt_masks, dim=1) - 1
        if action_in is not None:
            action_in = action_in.transpose(1, 0)
        obs_action_tokens, obs_action_masks, n_max_objs, token_length = prepare_single_obs_action_tokens(
            obs_tokens_to_forward.transpose(1, 0),
            obs_masks_to_forward.transpose(1, 0),
            action_in,
            policy
        )
        position_ids = torch.cumsum(obs_action_masks, dim=0) - 1
        position_ids = position_ids.long().transpose(1, 0)
        obs_action_tokens = obs_action_tokens.transpose(1, 0)
        obs_action_masks = obs_action_masks.transpose(1, 0)
        data_batch["obs_action_tokens"].append(obs_action_tokens)
        data_batch["obs_action_masks"].append(obs_action_masks)
        data_batch["prompt_tokens"].append(prompt_tokens)
        data_batch["prompt_masks"].append(prompt_masks)
        data_batch["position_ids"].append(position_ids)
        data_batch["prompt_position_ids"].append(prompt_position_ids)
        data_batch["n_max_objs"].append(n_max_objs)
        data_batch["target_actions"].append(target_action)
        data_batch["token_lengths"].append(token_length)
        data_batch["is_rotation"].append(is_rotation)
        data_batch["prompt"].append( traj['meta']['prompt'].lower())
        data_batch["task"].append( traj["task"] )
    data_to_pad = (
        "obs_action_tokens",
        "obs_action_masks",
        "prompt_tokens",
        "prompt_masks",
        "position_ids",
        "prompt_position_ids",
    )
    for feature_name in data_to_pad:
        data_batch[feature_name] = batch_level_padding(
            data_batch[feature_name], 1, device
        )
    data_batch: DataBatch
    return data_batch

def batch_level_padding(
        batch_to_pad: List[Tensor], 
        padding_dim: int, 
        device: Device) -> Tensor:
    max_L = max((t.shape[padding_dim] for t in batch_to_pad))
    for idx in range(len(batch_to_pad)):
        tensor_to_pad = batch_to_pad[idx]
        L_padding = max_L - tensor_to_pad.shape[padding_dim]
        if L_padding == 0:
            continue
        assert L_padding > 0
        shape = tensor_to_pad.shape
        padding_shape = [
            (shape[i] if i != padding_dim else L_padding) for i in range(len(shape))
        ]
        batch_to_pad[idx] = any_concat(
            [
                tensor_to_pad,
                torch.zeros(*padding_shape, device=device, dtype=tensor_to_pad.dtype)
            ],
            dim=padding_dim
        )
    return any_concat(batch_to_pad, dim=0)