from typing import Tuple, List
from playground.typing import (
    ActionTokenEmbedding,
    DecodedAction,
    ContinuousAction,
    DiscreteAction,
    VIMAPolicy,
    ActionBounds,
    Device
)
import numpy as np
import torch


def decode_last_step_action(
        predicted_action_tokens: ActionTokenEmbedding,
        policy: VIMAPolicy
    ) -> Tuple[DecodedAction, ActionTokenEmbedding]:
    return decode_nth_step_action(
        predicted_action_tokens,
        policy,
        -1,
        False,
    )


def discretize_action(
        action: ContinuousAction, 
        n_discrete_x_bins: int = 50, 
        n_discrete_y_bins: int = 100, 
        n_discrete_rot_bins: int = 50,
    ) -> DiscreteAction:
    assert torch.is_floating_point(action["pose0_position"])
    device = action["pose0_position"].device
    boundary_x = torch.linspace(
        start=0, end=1, steps=n_discrete_x_bins, device=device
    )
    boundary_y = torch.linspace(
        start=0, end=1, steps=n_discrete_y_bins, device=device
    )
    boundary_rot = torch.linspace(
        start=0, end=1, steps=n_discrete_rot_bins, device=device
    )

    action["pose0_position"][..., 0] = torch.bucketize(
        action["pose0_position"][..., 0].contiguous(), boundary_x
    )
    action["pose0_position"][..., 1] = torch.bucketize(
        action["pose0_position"][..., 1].contiguous(), boundary_y
    )
    action["pose0_rotation"] = torch.bucketize(
        action["pose0_rotation"].contiguous(), boundary_rot
    )

    action["pose1_position"][..., 0] = torch.bucketize(
        action["pose1_position"][..., 0].contiguous(), boundary_x
    )
    action["pose1_position"][..., 1] = torch.bucketize(
        action["pose1_position"][..., 1].contiguous(), boundary_y
    )
    action["pose1_rotation"] = torch.bucketize(
        action["pose1_rotation"].contiguous(), boundary_rot
    )
    action = {k: v.long() for k, v in action.items()}
    return action

def decode_nth_step_action(
        predicted_action_tokens: ActionTokenEmbedding,
        policy: VIMAPolicy,
        nth: int = 1,
        discretize: bool = False,
    ) -> Tuple[DecodedAction, ActionTokenEmbedding]:
    predicted_action_tokens = predicted_action_tokens[nth].unsqueeze(
        0
    )  # (1, B, E)
    dist_dict = policy.forward_action_decoder(predicted_action_tokens)
    actions = {k: v.mode() for k, v in dist_dict.items()}
    action_tokens = policy.forward_action_token(actions)  # (1, B, E)
    action_tokens = action_tokens.squeeze(0)  # (B, E)
    if not discretize:
        actions = policy._de_discretize_actions(actions)
    return actions, action_tokens



def bound_action_to_bin_action(
        actions: ContinuousAction, 
        action_bounds: List[ActionBounds], 
        device: Device,
        discrete: bool = False,
    ) -> DecodedAction:
    """
    given an action and action bound, normalize to bin (normalize to range 0 ~ 1)
    the bin value can then be descritize to long type
    """
    action_bounds_low = [
        action_bound["low"] for action_bound in action_bounds
    ]
    action_bounds_high = [
        action_bound["high"] for action_bound in action_bounds
    ]
    action_bounds_low = np.asarray(action_bounds_low)
    action_bounds_high = np.asarray(action_bounds_high)
    action_bounds_low = torch.tensor(
        action_bounds_low, dtype=torch.float32, device=device
    )
    action_bounds_high = torch.tensor(
        action_bounds_high, dtype=torch.float32, device=device
    )
    actions = {
        k: torch.tensor(data=v, dtype=torch.float32, device=device) for k, v in actions.items()
    }
    actions["pose0_rotation"] = torch.clamp(
        actions["pose0_rotation"], min=-1, max=1
    )
    actions["pose1_rotation"] = torch.clamp(
        actions["pose1_rotation"], min=-1, max=1
    )
    actions["pose0_rotation"] = (actions["pose0_rotation"] + 1) / 2
    actions["pose1_rotation"] = (actions["pose1_rotation"] + 1) / 2
    actions["pose0_position"] = torch.clamp(
        actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
    )
    actions["pose1_position"] = torch.clamp(
        actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
    )
    actions["pose0_position"] = ( ( actions["pose0_position"] - action_bounds_low ) 
                                 / (action_bounds_high - action_bounds_low) )
    actions["pose1_position"] = ( ( actions["pose1_position"] - action_bounds_low ) 
                                 / (action_bounds_high - action_bounds_low) )
    if discrete:
        actions = discretize_action(actions)
    return actions
    
   


def bin_action_to_bound_action(
        actions: ContinuousAction, 
        action_bounds: List[ActionBounds], 
        device: Device
    ) -> ContinuousAction:
    """
    inverse operation of bound_action_to_bin_action
    where 
    let b = action_bound, then
        a == bin_action_to_bound_action(bound_action_to_bin_action(a, b), b) for all valid a, b
    """
    assert torch.is_floating_point(actions["pose0_position"])
    action_bounds_low = [
        action_bound["low"] for action_bound in action_bounds
    ]
    action_bounds_high = [
        action_bound["high"] for action_bound in action_bounds
    ]
    action_bounds_low = np.asarray(action_bounds_low)
    action_bounds_high = np.asarray(action_bounds_high)
    action_bounds_low = torch.tensor(
        action_bounds_low, dtype=torch.float32, device=device
    )
    action_bounds_high = torch.tensor(
        action_bounds_high, dtype=torch.float32, device=device
    )
    actions["pose0_position"] = (
        actions["pose0_position"] * (action_bounds_high - action_bounds_low)
        + action_bounds_low
    )
    actions["pose1_position"] = (
        actions["pose1_position"] * (action_bounds_high - action_bounds_low)
        + action_bounds_low
    )
    actions["pose0_position"] = torch.clamp(
        actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
    )
    actions["pose1_position"] = torch.clamp(
        actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
    )
    actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
    actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
    actions["pose0_rotation"] = torch.clamp(
        actions["pose0_rotation"], min=-1, max=1
    )
    actions["pose1_rotation"] = torch.clamp(
        actions["pose1_rotation"], min=-1, max=1
    )
    return actions
