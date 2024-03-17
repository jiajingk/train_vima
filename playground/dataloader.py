from tensorflow.train import Example
import itertools
import numpy as np
import time
from collections import namedtuple
from dotenv import dotenv_values
import os
import boto3
import boto3.session
import io
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from torch.utils.data import DataLoader
import torch
from typing import (
    Tuple, 
    Union,
    Iterator, 
    Optional,
    List,  
    Dict, 
    Any,
)
from einops import rearrange
from playground.typing import (
    Traj,
    TrainParam,
    DatasetParam,
    TaskName
)
from torch.utils.data import Dataset


userdata = dotenv_values(".env")
os.environ["AWS_REGION"] = "ca-central-1"
os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")


def extract_s3_info(s3_url: str) -> Tuple[str, str]:
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme != 's3':
        raise ValueError("URL doesn't use the s3 scheme")
    bucket = parsed_url.netloc
    path = parsed_url.path.lstrip('/')
    return bucket, path


def read_object(s3_client: Any, s3_url) -> str:
    bucket_name, file_name = extract_s3_info(s3_url)
    tfrecord_bytes = io.BytesIO()
    s3_client.download_fileobj(bucket_name, file_name, tfrecord_bytes)
    tfrecord_bytes.seek(0)
    decoded_string = tfrecord_bytes.read()
    return decoded_string


def read_parallel_multithreading(keys: List[str]) -> Iterator[Optional[str]]:
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
    assert valid_end < total_data_size_per_task
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


def recursive_reconstruction(data: Union[Dict, bytes, List]) -> Dict:
    if isinstance(data, dict) and tuple(sorted(data.keys())) == ('data', 'type'):
        type_name = data['type'][0].decode('utf-8')
        if type_name == 'omegaconf.listconfig.ListConfig':
            size = len(data['data'])
            return [data['data'][str(i)][0].decode('utf-8') for i in range(size)]
        elif type_name == 'list':
            size = len(data['data'])
            return [
                recursive_reconstruction(data['data'][str(i)]) for i in range(size)
            ]
        elif type_name == 'tuple':
            size = len(data['data'])
            return tuple(
                [recursive_reconstruction(data['data'][str(i)]) for i in range(size)]
            )
        elif type_name == "<enum 'ProfilePedia'>":
            return data['data'][0]
        elif type_name == 'vimasim.tasks.components.encyclopedia.definitions.SizeRange':
            SizeRange = namedtuple('SizeRange', data['data'].keys())
            data['data']['high'] = tuple(data['data']['high'])
            data['data']['low'] = tuple(data['data']['low'])
            return SizeRange(**data['data'])
        raise ValueError(f"unable to continue with {data}")
    if isinstance(data, dict) and tuple(sorted(data.keys())) == ('data', 'dtype', 'shape'):
        shape = tuple(data['shape'])
        dtype = np.dtype(data['dtype'][0].decode('utf-8'))
        flatten_data = np.frombuffer(data['data'][0], dtype=dtype)
        return np.reshape(flatten_data, shape)
    if isinstance(data, dict):
        return {
            str_to_id(key): recursive_reconstruction(value)
                for key, value in data.items()
        }
    data = data[0]
    if isinstance(data, bytes):
        data_str = data.decode('utf-8')
        if data_str == 'None':
            return None
        return data_str
    return data


def unflatten_dict(flattened_dict: Dict, sep='-') -> Dict:
    unflattened_dict = {}
    for flat_key, value in flattened_dict.items():
        keys = flat_key.split(sep)
        d = unflattened_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return unflattened_dict


def str_to_id(key: str) -> int:
    if key.isdigit():
        return int(key)
    return key


def map_into_boundary(
        action, 
        bound
    ):
    lower_bound_dim_0 = bound['low'][0]
    upper_bound_dim_0 = bound['high'][0]
    lower_bound_dim_1 = bound['low'][1]
    upper_bound_dim_1 = bound['high'][1]


    pose0_position = action['pose0_position'].copy()
    pose0_position[:, 0] = (pose0_position[:, 0] - lower_bound_dim_0) / (upper_bound_dim_0 - lower_bound_dim_0)
    pose0_position[:, 1] = (pose0_position[:, 1] - lower_bound_dim_1) / (upper_bound_dim_1 - lower_bound_dim_1)


    pose1_position = action['pose1_position'].copy()
    pose1_position[:, 0] = (pose1_position[:, 0] - lower_bound_dim_0) / (upper_bound_dim_0 - lower_bound_dim_0)
    pose1_position[:, 1] = (pose1_position[:, 1] - lower_bound_dim_1) / (upper_bound_dim_1 - lower_bound_dim_1)


    lower_bound_rotation = -1
    upper_bound_rotation = 1


    pose0_rotation = action['pose0_rotation'].copy()
    pose0_rotation[:, 0] = (pose0_rotation[:, 0] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)
    pose0_rotation[:, 1] = (pose0_rotation[:, 1] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)
    pose0_rotation[:, 2] = (pose0_rotation[:, 2] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)
    pose0_rotation[:, 3] = (pose0_rotation[:, 3] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)


    pose1_rotation = action['pose1_rotation'].copy()
    pose1_rotation[:, 0] = (pose1_rotation[:, 0] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)
    pose1_rotation[:, 1] = (pose1_rotation[:, 1] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)
    pose1_rotation[:, 2] = (pose1_rotation[:, 2] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)
    pose1_rotation[:, 3] = (pose1_rotation[:, 3] - lower_bound_rotation) / (upper_bound_rotation - lower_bound_rotation)


    return {
        'pose0_position': pose0_position,
        'pose1_position': pose1_position,
        'pose0_rotation': pose0_rotation,
        'pose1_rotation': pose1_rotation,
    }


def deseralize(raw_record: str) -> Traj:
    example = Example.FromString(raw_record)
    result = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof('kind')
        result[key] = getattr(feature, kind).value
    reconstructed_result = recursive_reconstruction(unflatten_dict(result))
    reconstructed_result['meta']['action_bounds']['high'] = reconstructed_result['meta']['action_bounds']['high'][:-1]
    reconstructed_result['meta']['action_bounds']['low'] = reconstructed_result['meta']['action_bounds']['low'][:-1]
    reconstructed_result['action']['pose0_position'] = reconstructed_result['action']['pose0_position'][:, :-1]
    reconstructed_result['action']['pose1_position'] = reconstructed_result['action']['pose1_position'][:, :-1]
    reconstructed_result['obs']['rgb'] = {
        "front": rearrange(reconstructed_result['rgb_front'], 't h w c -> t c h w'),
        "top": rearrange(reconstructed_result['rgb_top'], 't h w c -> t c h w'),
    }
    return reconstructed_result


def load_batch(addrs, local_batch_size: int) -> List[str]:
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


class RecordDataset:
    def __init__(self, records: List[str]):
        self.records = records


    def __getitem__(self, idx: int) -> str:
        return self.records[idx]


    def __len__(self) -> int:
        return len(self.records)


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
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=local_batch_size, 
        sampler=train_sampler,
        shuffle=shuffle
    )
    if distributed:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        shuffle = False
    else:
        valid_sampler = None
        shuffle = True
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=local_batch_size, 
        sampler=valid_sampler,
        shuffle=shuffle
    )
    return train_loader, valid_loader


class TrajDataset(Dataset):


    def __init__(self, data):
        self.data = data


    def __getitem__(self, index):
        task_id, task_name = self.data[index]
        return str(task_id), task_name


    def __len__(self):
        return len(self.data)


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