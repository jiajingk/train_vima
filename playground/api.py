import numpy as np
from typing import Tuple, Optional
from vima.utils import any_stack, any_slice
from playground.typing import (
    ObsData,
    EnvHistory,
    VIMAPolicy,
    VIMAEnvBase,
    Device,
    ContinuousAction
)
from playground.util.prompt import create_inital_prompt, create_prompt_token
from playground.util.obs import get_obs_token_this_step, obs_obj_padding, prepare_obs_forward
from playground.util.action import decode_last_step_action, bin_action_to_bound_action


def debug_step(
        obs: ObsData, 
        env: VIMAEnvBase, 
        policy: VIMAPolicy,
        device: Device,
        history: Optional[EnvHistory] = None
    ) -> Tuple[ContinuousAction, EnvHistory]:
    if history is None:
        history = {
            "action_tokens": [],
            "obs_masks": [],
            "obs_token_embeddings": [],
            "prompt_mask": None,
            "prompt_token": None,
            "prompt": env.prompt,
            "prompt_asset": env.prompt_assets,
            "obs": []
        }
    if (history["prompt_mask"] is None) or (history["prompt_token"] is None):
        prompt_token_this_task, prompt_mask_this_task = create_prompt_token(
            *create_inital_prompt(env), 
            policy=policy, 
            device=device
        )
    else:
        prompt_token_this_task = history["prompt_token"] 
        prompt_mask_this_task = history["prompt_mask"]
    meta_info = env.meta_info
    obs_token_embedding_this_step, obs_mask_this_step = get_obs_token_this_step(obs, meta_info, policy, device)
    padded_obs_token_embeddings, padded_obs_masks = obs_obj_padding(
        history["obs_token_embeddings"] + [obs_token_embedding_this_step[0][0]],
        history["obs_masks"] + [obs_mask_this_step[0][0]],
        device
    )
    obs_token_embeddings_to_forward, padded_obs_masks_to_forward = prepare_obs_forward(
        padded_obs_token_embeddings, padded_obs_masks
    )
    if len(history["action_tokens"]) == 0:
        action_tokens_to_forward = None
    else:
        action_tokens_to_forward = any_stack(
            [any_stack(history["action_tokens"], dim=0)],
            dim=0,
        )
        action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
    predicted_action_tokens = policy.forward(
        obs_token=obs_token_embeddings_to_forward,
        obs_mask=padded_obs_masks_to_forward,
        action_token=action_tokens_to_forward,
        prompt_token=prompt_token_this_task,
        prompt_token_mask=prompt_mask_this_task,
    )
    actions, action_tokens = decode_last_step_action(predicted_action_tokens, policy)
    final_action = bin_action_to_bound_action(actions, [env.meta_info['action_bounds']], device)
    final_action = any_slice({k: v.cpu().numpy() for k, v in final_action.items()}, np.s_[0, 0])
    new_history = {
        "action_tokens": history["action_tokens"] + [action_tokens[0]],
        "obs_masks": history["obs_masks"] + [obs_mask_this_step[0][0]],
        "obs_token_embeddings": history["obs_token_embeddings"] + [obs_token_embedding_this_step[0][0]],
        "prompt_mask": prompt_mask_this_task,
        "prompt_token": prompt_token_this_task
    }
    return final_action, new_history


def step(
        obs: ObsData, 
        env: VIMAEnvBase, 
        policy: VIMAPolicy,
        device: Device,
        history: Optional[EnvHistory] = None
    ) -> Tuple[ContinuousAction, EnvHistory]:
    if history is None:
        history = {
            "action_tokens": [],
            "obs_masks": [],
            "obs_token_embeddings": [],
            "prompt_mask": None,
            "prompt_token": None
        }
    if (history["prompt_mask"] is None) or (history["prompt_token"] is None):
        prompt_token_this_task, prompt_mask_this_task = create_prompt_token(
            *create_inital_prompt(env), 
            policy=policy, 
            device=device
        )
    else:
        prompt_token_this_task = history["prompt_token"] 
        prompt_mask_this_task = history["prompt_mask"]
    meta_info = env.meta_info
    obs_token_embedding_this_step, obs_mask_this_step = get_obs_token_this_step(obs, meta_info, policy, device, is_train=False)
    padded_obs_token_embeddings, padded_obs_masks = obs_obj_padding(
        history["obs_token_embeddings"] + [obs_token_embedding_this_step[0][0]],
        history["obs_masks"] + [obs_mask_this_step[0][0]],
        device
    )
    obs_token_embeddings_to_forward, padded_obs_masks_to_forward = prepare_obs_forward(
        padded_obs_token_embeddings, padded_obs_masks
    )
    if len(history["action_tokens"]) == 0:
        action_tokens_to_forward = None
    else:
        action_tokens_to_forward = any_stack(
            [any_stack(history["action_tokens"], dim=0)],
            dim=0,
        )
        action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
    predicted_action_tokens = policy.forward(
        obs_token=obs_token_embeddings_to_forward,
        obs_mask=padded_obs_masks_to_forward,
        action_token=action_tokens_to_forward,
        prompt_token=prompt_token_this_task,
        prompt_token_mask=prompt_mask_this_task,
    )
    actions, action_tokens = decode_last_step_action(predicted_action_tokens, policy)
    final_action = bin_action_to_bound_action(actions, [env.meta_info['action_bounds']], device)
    final_action = any_slice({k: v.cpu().numpy() for k, v in final_action.items()}, np.s_[0, 0])
    return final_action, {
        "action_tokens": history["action_tokens"] + [action_tokens[0]],
        "obs_masks": history["obs_masks"] + [obs_mask_this_step[0][0]],
        "obs_token_embeddings": history["obs_token_embeddings"] + [obs_token_embedding_this_step[0][0]],
        "prompt_mask": prompt_mask_this_task,
        "prompt_token": prompt_token_this_task
    }