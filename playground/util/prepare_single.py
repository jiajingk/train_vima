import numpy as np
import torch
from typing import Tuple, Union, Dict, Optional, TypeVar
from playground.typing import (
    NormalizedTraj, 
    Device,
    VIMAPolicy, 
    Tensor, 
    BinAction,
    ObsData,
    SingleStepViewData
)
from vima.utils import (
    add_batch_dim,
    any_concat, 
    any_stack,
    DataDict
)
from einops import rearrange
from playground.util.obs import prepare_obs
from playground.util.prompt import prepare_prompt
from playground.util.traj import get_total_time_step
from copy import deepcopy

A = TypeVar('A')

def add_batch_dims(args: Tuple[A]) -> Tuple[A]:
    return tuple(
        add_batch_dim(arg) for arg in args
    )


def traj_time_at(
        traj: NormalizedTraj, 
        time_index: int, 
        except_keys: Tuple[str] = ('meta',)
    ) -> NormalizedTraj:
    def slice_at(
            data: Union[Dict, np.ndarray], 
            idx: int, 
            total_idx: int, 
            except_keys: Tuple[str], 
            key: Optional[str] = None
        ) -> Union[Dict, np.ndarray]:
        if key is not None and key in except_keys:
            return data
        if isinstance(data, np.ndarray):
            if len(data.shape) > 1 and data.shape[0] in (total_idx, total_idx + 1,):
                return data[idx]
        if isinstance(data, dict):
            return {
                key: slice_at(value, idx, total_idx, except_keys, key) for key, value in data.items()
            }
        return data
    total_steps = get_total_time_step(traj)
    return slice_at(traj, time_index, total_steps, except_keys)


def get_separate_obs_and_rgb(
        step_traj: NormalizedTraj, 
    ) -> Tuple[ObsData, SingleStepViewData]:
    obs = step_traj['obs']
    obs["ee"] = np.asarray(obs["ee"][0])
    obs = obs
    rgb_dict = {
            'front': rearrange(step_traj['obs']['rgb']['front'], 'c w h -> c w h'),
            'top': rearrange(step_traj['obs']['rgb']['top'], 'c w h -> c w h'),
    }
    obs.pop('rgb')
    return obs, rgb_dict


def prepare_single_record_obs(
        traj: NormalizedTraj, 
        device: Device, 
        policy: VIMAPolicy
    ) -> Tuple[Tensor, Tensor]:
    """
    return the observation with shape (1, L, Q, E)
           the observation mask with (1, L, Q)
    1 -> Batch size
    L -> Traj length
    Q -> num of object
    E -> embedding
    """
    total_step = get_total_time_step(traj)
    history_cache = {}
    history_cache["obs_tokens"] = []
    history_cache["obs_masks"] = []
    history_cache["action_tokens"] = []
    for t_step in range(total_step):
        step_traj = traj_time_at(deepcopy(traj), t_step)
        obs, rgb_dict = get_separate_obs_and_rgb(step_traj)
        obs, rgb_dict = add_batch_dims((obs, rgb_dict,))
        obs: DataDict = prepare_obs(
            obs=obs,
            rgb_dict=rgb_dict,
            meta=step_traj['meta']
        )
        obs = obs.to_torch_tensor(
            device=device
        )
        obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
        obs_token_this_step = obs_token_this_step.squeeze(0)
        obs_mask_this_step = obs_mask_this_step.squeeze(0)
        history_cache["obs_tokens"].append(obs_token_this_step[0])
        history_cache["obs_masks"].append(obs_mask_this_step[0])
        max_objs = max(x.shape[0] for x in history_cache["obs_tokens"])
        obs_tokens_to_forward, obs_masks_to_forward = [], []
        obs_tokens_this_env, obs_masks_this_env = [], []
        for idx in range(len(history_cache["obs_tokens"])):
            obs_this_env_this_step = history_cache["obs_tokens"][idx]
            obs_mask_this_env_this_step = history_cache["obs_masks"][idx]
            required_pad = max_objs - obs_this_env_this_step.shape[0]
            obs_tokens_this_env.append(
                any_concat(
                    [
                        obs_this_env_this_step,
                        torch.zeros(
                            required_pad,
                            obs_this_env_this_step.shape[1],
                            device=device,
                            dtype=obs_this_env_this_step.dtype,
                        ),
                    ],
                    dim=0,
                )
            )
            obs_masks_this_env.append(
                any_concat(
                    [
                        obs_mask_this_env_this_step,
                        torch.zeros(
                            required_pad,
                            device=device,
                            dtype=obs_mask_this_env_this_step.dtype,
                        ),
                    ],
                    dim=0,
                )
            )
        obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
        obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
        obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
        obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
    return obs_tokens_to_forward, obs_masks_to_forward


def prepare_single_record_optional_action(
        traj: NormalizedTraj, 
        device: Device, 
        policy: VIMAPolicy
    ) -> Tuple[Optional[Tensor], BinAction]:
    action = {
        key: torch.from_numpy(value).unsqueeze(0).to(device)
            for key, value in traj['action'].items()
    }
    action = policy.discretize_action(action)
    action_tokens = policy.forward_action_token(action)
    target_action = {k: v.clone().squeeze(0) for k, v in action.items()}
    L_action = action_tokens.shape[1]
    if L_action <= 1:
        action_in = None
    else:
        action_in = action_tokens[:, :-1, :].clone()
    return action_in, target_action


def prepare_single_record_prompt(
        traj: NormalizedTraj, 
        device: Device, 
        policy: VIMAPolicy
    ) -> Tuple[Tensor, Tensor]:
    """
    return the prompt tokens with shape (L, 1, E)
           the prompt mask with (1, L)
    1 -> Batch size
    L -> Token Length
    """
    image_batch: DataDict
    word_batch: Tensor
    prompt_token_type, word_batch, image_batch = prepare_prompt(
        prompt=traj['meta']['prompt'],
        prompt_assets=traj['meta']['prompt_assets'],
        views=traj['meta']['views']
    )
    word_batch = word_batch.to(device)
    image_batch = image_batch.to_torch_tensor(device=device)
    prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
        (prompt_token_type, word_batch, image_batch)
    )
    prompt_tokens = prompt_tokens.transpose(1, 0)
    return prompt_tokens, prompt_masks


def prepare_single_obs_action_tokens(
        obs_token: torch.Tensor,  # L, B, Q, E
        obs_mask: torch.Tensor,  # L, B, E
        action_token: Optional[torch.Tensor], # L, B, Q
        policy: VIMAPolicy
    ) -> Tuple[Tensor, Tensor, int, int]:
    L_obs, B = obs_token.shape[:2]
    L_action = 0 if action_token is None else action_token.shape[0]
    n_max_objs = obs_token.shape[-2]
    L = L_obs * n_max_objs + L_action
    tokens = torch.empty(
        L, B, policy.embed_dim, dtype=torch.float32, device=obs_token.device
    )
    masks = torch.ones(L, B, dtype=torch.bool, device=obs_token.device)
    obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
    obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
    obs_token = rearrange(obs_token, "B L E -> L B E")
    obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
    obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
    obs_mask = rearrange(obs_mask, "B L -> L B")
    for q in range(n_max_objs):
        tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]
        masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
    if action_token is not None:
        tokens[n_max_objs :: n_max_objs + 1] = action_token
    return tokens, masks, n_max_objs, L