from torch.utils.data import DataLoader
from typing import (
    Tuple, 
    Optional,
)
from einops import rearrange
from playground.typing import (
    TrainParam,
    DatasetParam,
)
from playground.source.s3 import get_s3_dataloader
from playground.source.local import get_local_dataloader


def get_dataloader(
        train_param: TrainParam,
        dataset_param: DatasetParam,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    if dataset_param['source'] == 'local':
        return get_local_dataloader(
            train_param,
            dataset_param
        )
    if dataset_param['source']  == 's3://vima':
        return get_s3_dataloader(
            train_param,
            dataset_param
        )
    raise AssertionError(f'{dataset_param["source"]} dataloader is not implemented')