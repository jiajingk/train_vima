from typing import List, Callable
from playground.typing import (
    StepMeasure,
    FlattenedStepMeasure,
    ForwardMetaData,
    ActionAxisWeight,
    TaskWeight,
    Tensor
)
from playground.util.measure import (
    get_default_flatten_step_measure, 
    to_flatten_step_measure
)
from playground.util.prompt import get_task_class
import torch


def reduce_traj_loss_in_time_axis(
        traj_loss: List[StepMeasure],
        time_weight_map: Callable[[int], float],
        normalize: bool = True
    ) -> FlattenedStepMeasure:
    assert len(traj_loss) > 0
    total_traj_loss = get_default_flatten_step_measure(0.0)
    for i, step_measure in enumerate(traj_loss):
        for key, value in to_flatten_step_measure(step_measure, as_float=False).items():
            total_traj_loss[key] += (value * time_weight_map(i))
    if not normalize:
        return total_traj_loss
    return {
        key: value / len(traj_loss) for key, value in total_traj_loss.items()
    }



def reduce_weighted_step_total_loss(
        traj_imitation_loss: FlattenedStepMeasure,
        forward_meta: ForwardMetaData,
        axis_weights: ActionAxisWeight,
        task_weights: TaskWeight
    ) -> Tensor:
    step_loss = 0.0
    task = get_task_class(forward_meta["prompt"])
    if task not in task_weights:
        task_weight = 1.0
    else:
        task_weight = task_weights[task]
    for dim_name, loss in traj_imitation_loss.items():
        step_loss = step_loss + axis_weights[dim_name] * loss
    return step_loss * task_weight


def reduce_sum_step_total_loss(
        step_imitation_loss: StepMeasure,
        forward_meta: ForwardMetaData,
        apply_rotation_mask: bool,
    ) -> Tensor:
    step_loss = 0.0
    for dim_name, axis_losses in step_imitation_loss.items():
        apply_mask = ('rotation' in dim_name) and (forward_meta["is_rotation"] is False) and apply_rotation_mask
        mask = 0.0 if apply_mask else 1.0
        step_loss += mask * torch.sum(torch.stack(axis_losses))
    return step_loss
