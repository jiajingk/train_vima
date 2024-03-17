from typing import Union
from playground.typing import (
    NormalizedTraj,
    Traj,
    ObsData,
    Modality,
    View
)
from vima.utils import any_slice
import numpy as np
from einops import rearrange


def get_total_time_step(traj: Union[NormalizedTraj, Traj]) -> int:
    return traj['meta']['steps']


def get_obs_at_time(
        traj: Union[NormalizedTraj, Traj],
        t_step: int
    ) -> ObsData:
    assert len(traj['obs']['rgb']['front'].shape) == 4
    assert traj['obs']['rgb']['front'].shape[1] == 3 # T C H W
    assert traj['obs']['rgb']['front'].shape[0] > t_step
    return {
        'rgb': any_slice(
            traj['obs']['rgb'], np.s_[t_step]
        ),
        'segm': any_slice(
            traj['obs']['segm'], np.s_[t_step]
        ),
        'ee': traj['obs']['ee']
    }


def get_view_at_time(
    traj: Union[NormalizedTraj, Traj],
    t_step: int,
    modal: Modality = 'rgb',
    view_point: View = 'top'
    ) -> np.ndarray:
    obs_at_t = get_obs_at_time(traj, t_step)
    view_data = obs_at_t[modal][view_point]
    if modal == 'rgb':
        return rearrange(
            view_data, 'c h w -> h w c'
        )
    if modal == 'segm':
        return rearrange(
            view_data, 'h w -> h w 1'
        )
    raise AssertionError('Expected code is unreachable')
