from typing import Tuple, Dict, List, TypeVar
from playground.typing import (
    Env, 
    ObsData, 
    Action, 
    TaskInfo, 
    TaskMetaData, 
    Traj, 
    PromptAsset, 
    EnvMetaInfo,
    OracleActionSpace, 
    OracleActionBound,
    NormalizedTraj,
    CutTraj,
    TaskName,
)
from playground.util.action import bound_action_to_bin_action
import os
import pickle
import numpy as np
from einops import rearrange
from vima.utils import any_slice
from PIL import Image
from copy import deepcopy


def load(path: str) -> Traj:
    assert os.path.exists(path), path
    assert os.path.exists(os.path.join(path, "obs.pkl"))
    assert os.path.exists(os.path.join(path, "action.pkl"))
    assert os.path.exists(os.path.join(path, "trajectory.pkl"))
    assert os.path.exists(os.path.join(path, "rgb_front"))
    assert os.path.exists(os.path.join(path, "rgb_top"))
    with open(os.path.join(path, "obs.pkl"), "rb") as f:
        obs = pickle.load(f)
    rgb_dict = {"front": [], "top": []}
    n_rgb_frames = len(os.listdir(os.path.join(path, f"rgb_front")))
    for view in ["front", "top"]:
        for idx in range(n_rgb_frames):
            # load {idx}.jpg using PIL
            rgb_dict[view].append(
                rearrange(
                    np.array(
                        Image.open(os.path.join(path, f"rgb_{view}", f"{idx}.jpg")),
                        copy=True,
                        dtype=np.uint8,
                    ),
                    "h w c -> c h w",
                )
            )
    rgb_dict = {k: np.stack(v, axis=0) for k, v in rgb_dict.items()}
    with open(os.path.join(path, "action.pkl"), "rb") as f:
        action = pickle.load(f)
    with open(os.path.join(path, "trajectory.pkl"), "rb") as f:
        traj_meta = pickle.load(f)
    obs['rgb'] = rgb_dict
    return {
        "action": action,
        "meta": traj_meta,
        "obs": obs
    }


class OracleAgent:
    def __init__(self, env: Env):
        self.env = env

    def act(self, obs: ObsData) -> Action:
        assert self.env.current_task is not None
        assert self.env.time_step >= 0
        if self.env.time_step >= self.env.current_task["action"]["pose0_position"].shape[0]:
            raise ValueError("Unable to provide action, because task already done")
        del obs
        return any_slice(
            self.env.current_task["action"], np.s_[self.env.time_step]
        )


class Task:
    def __init__(self):
        ...


    def oracle(self, env: Env) -> OracleAgent:
        return OracleAgent(env)


class ReplayEnv:
    def __init__(self, folder: str):
        self.time_step: int = -1
        self.task_step: int = -1
        self.global_seed: int = 0
        self.tasks_source_folder = folder
        if os.path.exists(os.path.join(folder, 'metadata.pkl')):
            with open(os.path.join(folder, 'metadata.pkl'), 'rb') as f:
                self.meta_data: TaskMetaData = pickle.load(f)
        else:
            self.meta_data = {'seed_max': 128}
        self.total_task = self.meta_data['seed_max'] + 1
        self.action_history: Dict[int, List[Action]] = {}


    @property
    def current_task(self) -> Traj:
        assert self.task_step >= 0
        assert self.task_step < self.total_task
        return load(
            os.path.join(self.tasks_source_folder, f'{self.task_step:06}')
        ) 


    @property
    def action_space(self) -> OracleActionSpace:
        assert self.current_task is not None
        position_low = self.current_task["meta_info"]["action_bounds"]["low"][0]
        position_high = self.current_task["meta_info"]["action_bounds"]["high"][0]
        rotation_low = self.current_task["meta_info"]["action_bounds"]["low"][1]
        rotation_high = self.current_task["meta_info"]["action_bounds"]["high"][1]
        return {
            "pose0_position": OracleActionBound(position_low, position_high),
            "pose0_rotation": OracleActionBound(rotation_low, rotation_high),
            "pose1_position": OracleActionBound(position_low, position_high),
            "pose1_rotation": OracleActionBound(rotation_low, rotation_high),
        }


    @property
    def task(self) -> Task:
        return Task()


    @property
    def meta_info(self) -> EnvMetaInfo:
        assert self.current_task is not None
        return self.current_task["meta_info"]


    @property
    def prompt(self) -> str:
        assert self.current_task is not None
        return self.current_task["meta_info"]["prompt"]


    @property
    def prompt_assets(self) -> PromptAsset:
        assert self.current_task is not None
        return self.current_task["meta_info"]["prompt_assets"]
    
    def set_traj(self, task_id: int) -> ObsData:
        self.time_step = 0
        self.task_step = task_id
        if self.task_step >= self.total_task:
            raise ValueError("No more tasks")
        self.action_history[self.task_step] = []
        return any_slice(self.current_task["obs"], np.s_[self.time_step])


    def reset(self) -> ObsData:
        self.time_step = 0
        self.task_step += 1
        if self.task_step >= self.total_task:
            raise ValueError("No more tasks")
        self.action_history[self.task_step] = []
        return any_slice(self.current_task["obs"], np.s_[self.time_step])


    def step(self, action: Action) -> Tuple[ObsData, int, bool, TaskInfo]: 
        assert self.time_step >= 0
        self.time_step += 1
        self.action_history[self.task_step].append(action)
        total_step = self.current_task["action"]["pose0_position"].shape[0]
        if self.time_step > total_step:
            raise ValueError("this task already done")
        if self.time_step == total_step:
            done = True
            reward = 1
        else:
            done = False
            reward = 0
        obs = any_slice(self.current_task["obs"], np.s_[self.time_step])
        return obs, done, reward, {
            "failure": False,
            "success": done,
            "prompt": self.current_task["meta_info"]["prompt"],
            "TimeLimit.truncated": False
        }


def normalize_traj(traj: Traj) -> NormalizedTraj:
    actions = traj['action']
    normalized_traj: NormalizedTraj = {
        **traj,
        "action": bound_action_to_bin_action(actions, [traj['meta']['action_bounds']], 'cpu')
    }
    normalized_traj['action'] = {
        k: v.cpu().numpy() for k, v in normalized_traj['action'].items()
    }
    return normalized_traj

def load_traj(task_id: int, task_name: TaskName = 'rotate') -> Traj:
    env = ReplayEnv(os.path.join('tasks', task_name))
    env.set_traj(task_id)
    return deepcopy(env.current_task)

def load_trajs(amount: int = 128, task_name: TaskName = 'rotate') -> List[Traj]:
    env = ReplayEnv(os.path.join('tasks', task_name))
    trajs = []
    for _ in range(amount):
        env.reset()
        trajs.append(deepcopy(env.current_task))
    return trajs

def load_normalized_trajs(amount: int = 128, task_name: str = 'rotate') -> List[NormalizedTraj]:
    return [
        normalize_traj(traj) for traj in load_trajs(amount, task_name)
    ]

def load_normalized_traj(task_id: int, task_name: str = 'rotate') -> NormalizedTraj:
    return normalize_traj(load_traj(task_id, task_name)) 

TrajType = TypeVar('TrajType', Traj, NormalizedTraj)

def cut_at_time(time_index: int, traj: Traj) -> CutTraj:
    assert len(traj["action"]["pose0_position"].shape) == 2
    assert len(traj["obs"]["rgb"]["front"].shape) == 4
    assert traj["obs"]["rgb"]["front"].shape[0] - 1 == traj["action"]["pose0_position"].shape[0]
    assert time_index > 0
    return {
        **traj,
        "action": any_slice(traj["action"], np.s_[:time_index-1]) if time_index - 1 > 0 else None,
        "obs": any_slice(traj["obs"], np.s_[:time_index])
    } 