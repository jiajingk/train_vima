from playground.typing import (
    DatasetParam
)


def get_dataset_param() -> DatasetParam:
    return  {
        "data_pct_usage": 0.5,
        "total_data_size_per_task": 40000,
        "validation_pct": 0.1,
        "source": "s3://vima",
        "tasks": [
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
