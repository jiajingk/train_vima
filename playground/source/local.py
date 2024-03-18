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
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    per_task_data = 128
    task_used = dataset_param['tasks']
    all_ids = range(per_task_data)
    train_samples = [
        (i, task) for task, i in itertools.product(
            task_used, all_ids
        )
    ]
    train_samples = [
        (i, task,) for task, i in itertools.product(
            task_used, all_ids
        )
    ]
    dataset = TrajDataset(train_samples)
    return DataLoader(dataset, batch_size=train_param['local_batch_size'], shuffle=True), None