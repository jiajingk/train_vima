
import itertools
import time
import os
import boto3
import boto3.session
import io
import torch

from dotenv import dotenv_values
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import (
    Tuple,  
    Optional,
    List,
    Any,
    Iterator
)
from playground.typing import (
    TrainParam,
    DatasetParam,
    TaskName
)

userdata = dotenv_values(".env")
os.environ["AWS_REGION"] = "ca-central-1"
os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")


class RecordDataset:
    def __init__(self, records: List[str]):
        self.records = records


    def __getitem__(self, idx: int) -> str:
        return self.records[idx]


    def __len__(self) -> int:
        return len(self.records)


def prepare_task_id(
        data_pct_usage: float = 0.85,
        validation_pct: float = 0.10,
        total_data_size_per_task: int = 40000
    ) -> Tuple[List[int], List[int]]:
    train_start = 0
    train_end = int(total_data_size_per_task * data_pct_usage)
    train_task_id = [
        i for i in range(train_start, train_end)
    ]
    train_size = (train_end - train_start)
    valid_start = train_end
    valid_end = valid_start + int(validation_pct * train_size)
    assert valid_end <= total_data_size_per_task
    valid_task_id = [
        i for i in range(valid_start,  valid_end)
    ]
    return train_task_id, valid_task_id


def get_records_addr(
        data_pct_usage: float = 0.85,
        validation_pct: float = 0.10,
        total_data_size_per_task: int = 40000,
        tasks: Optional[List[TaskName]] = None
    ) -> Tuple[List[str], List[str]]:
    if tasks is None:
        task_used = [
            'visual_manipulation',
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
    else:
        task_used = tasks
    train_task_id, valid_task_id = prepare_task_id(
        data_pct_usage,
        validation_pct,
        total_data_size_per_task
    )
    task_used
    train_records = [
        f's3://vima-tfrecords/{task}-{i}.tfrecord' for task, i in itertools.product(
            task_used, train_task_id
        )
    ]
    valid_records = [
        f's3://vima-tfrecords/{task}-{i}.tfrecord' for task, i in itertools.product(
            task_used, valid_task_id
        )
    ]
    return train_records, valid_records


def get_train_valid_data_loader(
        data_pct_usage: float,
        validation_pct: float,
        total_data_size_per_task: int,
        local_batch_size: int,
        distributed: bool = False,
        tasks: Optional[List[TaskName]] = None 
    ) -> Tuple[DataLoader, DataLoader]:
    train_dataset_address, valid_dataset_address = get_records_addr(
        data_pct_usage = data_pct_usage,
        validation_pct = validation_pct,
        total_data_size_per_task = total_data_size_per_task,
        tasks = tasks
    )
    train_dataset = RecordDataset(train_dataset_address)
    valid_dataset = RecordDataset(valid_dataset_address)
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset, 
        batch_size=local_batch_size, 
        sampler=train_sampler,
        shuffle=shuffle
    )
    if distributed:
        valid_sampler = DistributedSampler(valid_dataset)
        shuffle = False
    else:
        valid_sampler = None
        shuffle = True
    if len(valid_dataset_address) == 0:
        valid_loader = None
    else:
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=local_batch_size, 
            sampler=valid_sampler,
            shuffle=shuffle
        )
    return train_loader, valid_loader


def get_s3_dataloader(
        train_param: TrainParam,
        dataset_param: DatasetParam
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    return get_train_valid_data_loader(
        dataset_param['data_pct_usage'],
        dataset_param['validation_pct'],
        dataset_param['total_data_size_per_task'],
        train_param['local_batch_size'],
        train_param['distributed'],
        dataset_param['tasks']
    )

def extract_s3_info(s3_url: str) -> Tuple[str, str]:
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme != 's3':
        raise ValueError("URL doesn't use the s3 scheme")
    bucket = parsed_url.netloc
    path = parsed_url.path.lstrip('/')
    return bucket, path


def read_object(s3_client: Any, s3_url: str) -> Tuple[str, str]:
    bucket_name, file_name = extract_s3_info(s3_url)
    tfrecord_bytes = io.BytesIO()
    s3_client.download_fileobj(bucket_name, file_name, tfrecord_bytes)
    tfrecord_bytes.seek(0)
    decoded_string = tfrecord_bytes.read()
    return (decoded_string, s3_url.split('/')[-1].split('-')[0])


def read_parallel_multithreading(keys: List[str]) -> Iterator[Optional[Tuple[str, str]]]:
    session = boto3.session.Session()
    s3_client = session.client("s3")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_key = {executor.submit(read_object, s3_client, key): key for key in keys}
        for future in futures.as_completed(future_to_key):
            exception = future.exception()
            if not exception:
                yield future.result()
            else:
                yield None


S3_ADDR = str
TF_RECORD_STR = str

def load_batch(
        addrs: List[S3_ADDR], 
        local_batch_size: int
    ) -> List[TF_RECORD_STR]:
    retry = 0
    max_retry = 20
    while retry < max_retry:
        try:
            return [ 
                data for data in read_parallel_multithreading(addrs[:local_batch_size]) if data is not None
            ]
        except Exception as e:
            print("failed on current batch due to")
            print(e)
            retry += 1
            time.sleep(retry * 5)
    return []