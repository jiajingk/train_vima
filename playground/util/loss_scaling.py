import math
import numpy as np
from typing import List, Dict, Callable
from playground.typing import (
    StepMeasure,
    ActionAxisWeight,
    ActionWeightMethod,
    ForwardMetaData,
    TaskWeight,
    ActionWeightMethod,
    TaskWeightMethod
)
from playground.util.measure import (
    average_step_measure,
    to_flatten_step_measure,
)
from playground.util.prompt import get_task_class


def get_default_task_weight(default_value: float = 1.0) -> TaskWeight:
    return {
        "follow_order": default_value,
        "manipulate_old_neighbor": default_value,
        "novel_adj": default_value,
        "novel_noun": default_value,
        "pick_in_order_then_restore": default_value,
        "rearrange": default_value,
        "rearrange_then_restore": default_value,
        "same_profile": default_value,
        "rotate": default_value,
        "scene_understanding": default_value,
        "simple_manipulation": default_value,
        "sweep_without_exceeding": default_value,
        "twist": default_value,
        "visual_manipulation": default_value,
    }


def get_default_axis_weight(default_value: float = 1.0) -> ActionAxisWeight:
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


def conditional_selective_normalization(
        data: Dict[str, float], 
        selector: Callable[[List[float]], float],
        condition:  Callable[[float], bool],
    ) -> Dict[str, float]:
    selected_value = selector(list(data.values()))
    normalized_data = {
        key: (value / selected_value if condition(value) else value)
        for key, value in data.items()
    }  
    return normalized_data


def get_action_weigts(
        batch_traj_measure: List[List[StepMeasure]],
        method: ActionWeightMethod
    ) -> ActionAxisWeight:
    if method == "scale_to_same_order_of_magnitude":
        return scale_to_same_order_of_magnitude(batch_traj_measure)
    elif method == "default":
        return get_default_axis_weight(1.0)
    elif method == "constant_scaling":
        weight = get_default_axis_weight(1.0)
        weight["pose0_position_1"] = math.log(50) / math.log(100)
        weight["pose1_position_1"] = math.log(50) / math.log(100)
        return weight
    raise AssertionError(f"{method} not implemented")

def scale_with_respect_to_the_batch_avg(
        batch_measure: List[List[StepMeasure]],
        batch_forward_meta: List[ForwardMetaData],
        selector: Callable[[List[float]], float],
        condition:  Callable[[float], bool],
    ) -> TaskWeight:
    accumulator = get_default_task_weight(0.0)
    for step_measures, forward_meta in zip(batch_measure, batch_forward_meta):
        traj_loss = average_step_measure(
            [to_flatten_step_measure(step_measure) for step_measure in step_measures]
        )
        accumulator[get_task_class(forward_meta["prompt"])] += sum(traj_loss.values())
    accumulator = conditional_selective_normalization(
        accumulator,
        selector,
        condition
    )
    return {
        key: 1.0 / value if value != 0.0 else 0.0 
            for key, value in accumulator.items()
    }


def get_task_weights(
        batch_measure: List[List[StepMeasure]],
        batch_forward_meta: List[ForwardMetaData],
        method: TaskWeightMethod
    ) -> TaskWeight:
    if method == "scale_with_respect_to_the_max_batch_avg_loss":
        def selector(data: List[float]) -> float:
            return max(sample for sample in data if sample > 0.0)
        def condition(sample: float) -> bool:
            return sample > 0.0
        return scale_with_respect_to_the_batch_avg(
            batch_measure, 
            batch_forward_meta,
            selector,
            condition  
        )
    elif method == "scale_with_respect_to_the_min_batch_avg_loss":
        def selector(data: List[float]) -> float:
            return min(sample for sample in data if sample > 0.0)
        def condition(sample: float) -> bool:
            return sample > 0.0
        return scale_with_respect_to_the_batch_avg(
            batch_measure, 
            batch_forward_meta,
            selector,
            condition  
        )
    elif method == "default":
        return get_default_task_weight(1.0)
    raise AssertionError(f"{method} not implemented")

def calculate_action_weight_factor(
        action_weight: ActionAxisWeight
    ) -> float:
    zero_count = list(action_weight.values()).count(0.0)
    total_weight_count = len(action_weight)
    if zero_count == len(action_weight):
        return 1.0
    return total_weight_count / (total_weight_count - zero_count)


def scale_to_same_order_of_magnitude(
        batch_traj_measure: List[List[StepMeasure]]
    ) -> ActionAxisWeight:
    batch_mean_measure = average_step_measure( 
        [
            average_step_measure(
                [to_flatten_step_measure(step_measure) for step_measure in traj_measure]
            ) for traj_measure in batch_traj_measure 
        ] 
    )
    max_abs_log10 = max([abs(num) for num in batch_mean_measure.values()], key=abs)
    target_magnitude = math.floor(math.log10(max_abs_log10))
    def get_scale_factor(number_to_scale: float) -> float:
        if number_to_scale == 0.0:
            return 0.0
        number_magnitude = math.log10(abs(number_to_scale))
        scale_factor = target_magnitude - math.floor(number_magnitude)
        return 10 ** scale_factor
    return {
        key: get_scale_factor(value) for key, value in batch_mean_measure.items()
    }