from tensorflow.train import Example
import numpy as np
from collections import namedtuple
from typing import ( 
    Union,
    List,  
    Dict, 
)
from einops import rearrange
from playground.typing import (
    Traj,
)



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