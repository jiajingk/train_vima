import itertools
from torch.utils.data import DataLoader
from typing import (
    Tuple,
    Optional,
)
from playground.typing import (
    TrainParam,
    DatasetParam,
)
from torch.utils.data import Dataset


class TrajDataset(Dataset):
    def __init__(self, data):
        self.data = data


    def __getitem__(self, index):
        task_id, task_name = self.data[index]
        return str(task_id), task_name


    def __len__(self):
        return len(self.data)


def get_local_dataloader(
        train_param: TrainParam,
        dataset_param: DatasetParam
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
    sample_ratio = dataset_param['data_pct_usage']
    assert sample_ratio <= 1.0
    per_task_data = int(dataset_param['total_data_size_per_task'] * sample_ratio)
    task_used = dataset_param['tasks']
    valid_ratio = dataset_param['validation_pct']
    assert 0 <= valid_ratio < 1.0
    split_index = int(per_task_data * (1 - valid_ratio))
    train_samples = [
        (i, task) for task, i in itertools.product(
            task_used, range(0, split_index)
        )
    ]
    valid_samples = [
        (i, task,) for task, i in itertools.product(
            task_used, range(split_index, per_task_data)
        )
    ]
    train_dataset = TrajDataset(train_samples)
    valid_dataset = TrajDataset(valid_samples) if len(valid_samples) > 0 else None
    return (
        DataLoader(train_dataset, batch_size=train_param['local_batch_size'], shuffle=True), 
        DataLoader(valid_dataset, batch_size=train_param['local_batch_size'], shuffle=True) if valid_dataset is not None else None
    )