from typing import List, Optional, Tuple
from playground.typing import (
    Tensor,
    StepMeasure,
    FlattenedStepMeasure,
    PredDist,
    DiscreteAction,
    PredDistMapper,
    ActionMapper,
    ForwardMetaData,
    Criterion,
    Action,
)


def get_default_flatten_step_measure(
        default_value: float = 1.0
    ) -> FlattenedStepMeasure:
    return {
        "pose0_position_0": default_value,
        "pose0_position_1": default_value,

        "pose1_position_0": default_value,
        "pose1_position_1": default_value,

        "pose0_rotation_0": default_value,
        "pose0_rotation_1": default_value,
        "pose0_rotation_2": default_value,
        "pose0_rotation_3": default_value,

        "pose1_rotation_0": default_value,
        "pose1_rotation_1": default_value,
        "pose1_rotation_2": default_value,
        "pose1_rotation_3": default_value,
    }




def measure_step_metrics(
        pred_dist: PredDist,
        target_action: DiscreteAction,
        criterion: Criterion,
        extract_logits: PredDistMapper,
        extract_target: ActionMapper,
        t_step: int
    ) -> StepMeasure:
    axis_mapping = {
        'pose0_position': (0, 1),
        'pose1_position': (0, 1),
        'pose0_rotation': (0, 1, 2, 3),
        'pose1_rotation': (0, 1, 2, 3),
    }
    step_imitation_loss: StepMeasure = {
        dim_name: [
            criterion(
                extract_logits(pred_dist, (dim_name, axis, t_step,)),
                extract_target(target_action, (dim_name, axis, t_step,))
            ) for axis in dims
        ] for dim_name, dims in axis_mapping.items()
    }
    return step_imitation_loss

def rotation_masked_measure_traj_metrics(
        pred_dist: PredDist, 
        target_action: DiscreteAction, 
        forward_meta: ForwardMetaData,
        criterion: Criterion,
        pred_mapper: PredDistMapper,
        target_mapper: ActionMapper,
    ) -> List[StepMeasure]:
    total_time_step = forward_meta["action_length"]
    traj_metrics_measure: List[StepMeasure] = [
        measure_step_metrics(
            pred_dist, target_action, criterion, pred_mapper, target_mapper, t_step
        ) for t_step in range(total_time_step)
    ]
    if forward_meta["task"].lower() in ("twist", "rotate"):
        return traj_metrics_measure
    for step_measure in traj_metrics_measure:
        for action_type in step_measure:
            if 'rotation' in action_type:
                for i in range(len(step_measure[action_type])):
                    step_measure[action_type][i] *= 0.0
    return traj_metrics_measure


def measure_traj_metrics(
        pred_dist: PredDist, 
        target_action: DiscreteAction, 
        forward_meta: ForwardMetaData,
        criterion: Criterion,
        pred_mapper: PredDistMapper,
        target_mapper: ActionMapper,
    ) -> List[StepMeasure]:
    total_time_step = forward_meta["action_length"]
    traj_metrics_measure: List[StepMeasure] = [
        measure_step_metrics(
            pred_dist, target_action, criterion, pred_mapper, target_mapper, t_step
        ) for t_step in range(total_time_step)
    ]
    return traj_metrics_measure


def measure_traj_accu(
        pred_dist: PredDist, 
        target_action: DiscreteAction, 
        forward_meta: ForwardMetaData
    ) -> List[StepMeasure]:
    def criterion(x: Tensor, y: Tensor) -> float:
        if abs(x - y) == 0:
            return 1
        else:
            return 0
    def extract_mode(  
            pred_dist: PredDist, 
            args: Tuple,
        ) -> Tensor:
        dim_name, axis, t_step = args
        dim_name: str
        axis: int
        t_step: int
        return pred_dist[dim_name]._dists[axis].mode()[t_step]
    def extract_action(
            action: Action,
            args: Tuple,
        ) -> Tensor:
        dim_name, axis, t_step = args
        dim_name: str
        axis: int
        t_step: int
        return action[dim_name][t_step][axis].long()
    total_time_step = forward_meta["action_length"]
    traj_imitation_loss: List[StepMeasure] = [
        measure_step_metrics(
            pred_dist, target_action, criterion, extract_mode, extract_action, t_step, 
        ) for t_step in range(total_time_step)
    ]
    return traj_imitation_loss


def measure_traj_individual_loss(
        pred_dist: PredDist, 
        target_action: DiscreteAction, 
        forward_meta: ForwardMetaData,
        criterion: Criterion,
        rotation_mask: bool = False
    ) -> List[StepMeasure]:
    def extract_logits(
            pred_dist: PredDist, 
            args: Tuple
        ) -> Tensor:
        dim_name, axis, t_step = args
        dim_name: str
        axis: int
        t_step: int
        return pred_dist[dim_name]._dists[axis].logits[t_step]
    def extract_action(
            action: Action,
            args: Tuple
        ) -> Tensor:
        dim_name, axis, t_step = args
        dim_name: str
        axis: int
        t_step: int
        return action[dim_name][t_step][axis].long()
    if rotation_mask is True:
        return rotation_masked_measure_traj_metrics(
            pred_dist, 
            target_action, 
            forward_meta, 
            criterion, 
            extract_logits, 
            extract_action
        )
    return measure_traj_metrics(
        pred_dist, 
        target_action, 
        forward_meta, 
        criterion, 
        extract_logits, 
        extract_action
    )
    


def to_flatten_step_measure(step_measure: StepMeasure, as_float: bool = True) -> FlattenedStepMeasure:
    default_measure = get_default_flatten_step_measure(0.0)
    for dim_name, measure in step_measure.items():
        for i, step_measure in enumerate(measure):
            if as_float:
                if isinstance(step_measure, Tensor):
                    step_measure = float(step_measure.item())
                elif step_measure is None:
                    step_measure = step_measure
                else:
                    step_measure = float(step_measure)
            default_measure[f'{dim_name}_{i}'] = step_measure
    return default_measure


def average_step_measure(
        flatten_step_measures: List[FlattenedStepMeasure]
    ) -> FlattenedStepMeasure:
    assert len(flatten_step_measures) > 0
    result = get_default_flatten_step_measure(0.0)
    for flatten_step_measure in flatten_step_measures:
        for dim_axis, value in flatten_step_measure.items():
            result[dim_axis] += value
    return {
        key: value / len(flatten_step_measures) for key, value in result.items()
    }
