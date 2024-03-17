import torch
import numpy as np
from playground.typing import ActionBounds, ContinuousAction, Device
from playground.util.action import (
    bin_action_to_bound_action,
    bound_action_to_bin_action
)


def test_bound_to_bin_mapping():
    device: Device = 'cpu'
    action: ContinuousAction = {
        "pose0_position": torch.Tensor([[[-0.2, 0.2]]]),
        "pose1_position": torch.Tensor([[[-0.1, 0.1]]]),
        "pose0_rotation": torch.Tensor([[[0.0, 0.1, -0.1, 1.0]]]),
        "pose1_rotation": torch.Tensor([[[0.0, -0.1, 0.1, -1.0]]]),
    }
    action_bounds: ActionBounds = {
        "low": np.array([-0.5, -0.5]),
        "high": np.array([0.5, 0.5]),
    }
    bin_action = bound_action_to_bin_action(
        action, [action_bounds], device, False
    )
    for v in bin_action.values():
        assert torch.max(v[... , 0]) <= 1.0
        assert torch.max(v[... , 1]) <= 1.0
        assert torch.min(v[... , 0]) >= 0.0
        assert torch.min(v[... , 1]) >= 0.0
    recovered_action = bin_action_to_bound_action(
        bin_action, [action_bounds], device
    )
    for k in action.keys():
        np.testing.assert_almost_equal(action[k].cpu().numpy(), recovered_action[k].cpu().numpy())


def test_bound_to_bin_mapping_2d():
    device: Device = 'cpu'
    action: ContinuousAction = {
        "pose0_position": torch.Tensor([[[-0.2, 0.2], [-0.3, 0.1]]]),
        "pose1_position": torch.Tensor([[[-0.1, 0.1], [-0.1, 0.22]]]),
        "pose0_rotation": torch.Tensor([[[0.0, 0.1, -0.1, 1.0], [0.0, -0.3, 0.7, -1.0]]]),
        "pose1_rotation": torch.Tensor([[[0.0, -0.1, 0.1, -1.0], [0.5, 0.1, -0.1, 1.0]]]),
    }
    
    action_bounds: ActionBounds = {
        "low": np.array([-0.5, -0.5]),
        "high": np.array([0.5, 0.5]),
    }
    bin_action = bound_action_to_bin_action(
        action, [action_bounds], device, False
    )
    for v in bin_action.values():
        assert torch.max(v[... , 0]) <= 1.0
        assert torch.max(v[... , 1]) <= 1.0
        assert torch.min(v[... , 0]) >= 0.0
        assert torch.min(v[... , 1]) >= 0.0
    recovered_action = bin_action_to_bound_action(
        bin_action, [action_bounds], device
    )
    for k in action.keys():
        np.testing.assert_almost_equal(action[k].cpu().numpy(), recovered_action[k].cpu().numpy())


def test_bound_to_bin_mapping_3d():
    device: Device = 'cpu'
    action: ContinuousAction = {
        "pose0_position": torch.Tensor([[[-0.2, 0.2], [-0.3, 0.1]],[[0.1, -0.35], [-0.13, 0.12]]]),
        "pose1_position": torch.Tensor([[[-0.1, 0.1], [-0.1, 0.22]],[[-0.25, -0.43], [0.33, 0.17]]]),
        "pose0_rotation": torch.Tensor([[[0.0, 0.1, -0.1, 1.0], [0.0, -0.3, 0.7, -1.0]],[[0.9, 0.1, -0.1, 1.0], [0.83, -0.3, 0.79, -0.95]]]),
        "pose1_rotation": torch.Tensor([[[0.0, -0.1, 0.1, -1.0], [0.5, 0.1, -0.1, 1.0]],[[-0.0, -0.1, -0.1, -1.0], [-0.68, -0.3, 0.37, -0.02]]]),
    }
    for k, v in action.items():
        print(k, v.shape)
    
    action_bounds: ActionBounds = {
        "low": np.array([-0.5, -0.5]),
        "high": np.array([0.5, 0.5]),
    }
    bin_action = bound_action_to_bin_action(
        action, [action_bounds], device, False
    )
    for v in bin_action.values():
        assert torch.max(v[... , 0]) <= 1.0
        assert torch.max(v[... , 1]) <= 1.0
        assert torch.min(v[... , 0]) >= 0.0
        assert torch.min(v[... , 1]) >= 0.0
    recovered_action = bin_action_to_bound_action(
        bin_action, [action_bounds], device
    )
    for k in action.keys():
        np.testing.assert_almost_equal(action[k].cpu().numpy(), recovered_action[k].cpu().numpy())