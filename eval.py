import torch
import numpy as np
import pandas as pd
import os
from typing import (
    Literal,
    List,
    Tuple,
    Dict,
    TypedDict
)
from playground.typing import (
    RunConfig, 
    EnvConfig, 
    TaskName,
    TaskLevel, 
    TaskInfo,
    Action
)
from playground.util.action import bound_action_to_bin_action
from playground.util.measure import to_flatten_step_measure
from playground.util.log import flatten_dict
from playground.util.policy import create_policy_from_ckpt
from playground.env import create_env, oracle_step, MakeConfig
from vima_bench import PARTITION_TO_SPECS, ALL_TASKS
from playground.api import step
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import argparse
import time
np.set_printoptions(2)

OracleAction = Action
PolicyAction = Action
ActionTrace = List[Tuple[OracleAction, PolicyAction]]
PerfTrace = List[Tuple[int, float]]

class FlattenedEvalRecord(TypedDict):
    action_trace: Dict[str, float]
    task: str
    prompt: str
    sucess: Literal[0, 1]
    failure: Literal[0, 1]
    timeout: Literal[0, 1]
    total_token: int
    total_time: float

class EvalRecord(TypedDict):
    seed: int
    action_trace: ActionTrace 
    perf_trace: PerfTrace
    task: str
    prompt: str
    step_count: int
    sucess: Literal[0, 1]
    failure: Literal[0, 1]
    timeout: Literal[0, 1]
    bad_action: Literal[0, 1]

@torch.no_grad()
def main(model_path: str, task: TaskName, count: int):
    run_config = RunConfig(
        partition="placement_generalization",
        task=task,
        ckpt=model_path,
        device="cuda"
    )
    env_config = EnvConfig(
        use_time_wrapper=True,
        use_reset_wrapper=True,
        make_config=MakeConfig(modalities=["segm", "rgb"],
        task_kwargs=PARTITION_TO_SPECS["test"][run_config.partition][run_config.task],
        seed=42,
        render_prompt=True,
        display_debug_window=True,
        hide_arm_rgb=False)
    )
    if 'M' in model_path:
        prefix = ''
    else:
        prefix = 'module.'
    policy = create_policy_from_ckpt(
        run_config.ckpt, 
        run_config.device, 
        prefix=prefix
    )
    policy.eval()
    env = create_env(run_config, env_config)
    oracle_agent = env.task.oracle(env)
    for i in range(count):
        env.seed(i)
        
        obs = env.reset()
        history = None
        while True:
            policy_action, history = step(obs, env, policy, run_config.device, history)
            oracle_action = oracle_step(obs, env, oracle_agent)
            if oracle_action is None:
                oracle_action = {
                    "pose0_position": None,
                    "pose1_position": None,
                    "pose0_rotation": None,
                    "pose1_rotation": None
                }
            print(f'{oracle_action["pose0_position"] = }')
            print(f'{policy_action["pose0_position"] = }')
            print("==================")
            print(f'{oracle_action["pose1_position"] = }')
            print(f'{policy_action["pose1_position"] = }')
            print("==================")
            print(f'{oracle_action["pose0_rotation"] = }')
            print(f'{policy_action["pose0_rotation"] = }')
            print("==================")
            print(f'{oracle_action["pose1_rotation"] = }')
            print(f'{policy_action["pose1_rotation"] = }')
            obs, _, done, info = env.step(policy_action)
            print(i, info)
            if done:
                break



def fill_traces(eval_records: List[EvalRecord]) -> List[FlattenedEvalRecord]:
    fill_num = 10
    none_action = {
        "pose0_position": [None, None],
        "pose1_position": [None, None],
        "pose0_rotation": [None, None, None, None],
        "pose1_rotation": [None, None, None, None]
    }
    def fill(eval_record: EvalRecord, fill_num: int) -> EvalRecord:
        return {
            **eval_record,
            "action_trace": (
                [action_step for action_step in eval_record["action_trace"]] + 
                [
                    (
                        deepcopy(none_action), 
                        deepcopy(none_action),
                    ) 
                    for _ in range(fill_num - len(eval_record["action_trace"]))
                ]
            )
        }
    def flatten_trace(eval_record: EvalRecord) -> FlattenedEvalRecord:
        perf_trace = eval_record.pop("perf_trace")
        return {
            **eval_record,
            "action_trace": flatten_dict({
                str(i): {
                    "oracle_action": to_flatten_step_measure(oracle_action),
                    "policy_action": to_flatten_step_measure(policy_action)
                } for i, (oracle_action, policy_action) in enumerate(
                    eval_record["action_trace"]
                )
            }),
            "total_token": sum([x[0] for x in perf_trace]),
            "total_time": sum([x[1] for x in perf_trace])
        }
    return [
        flatten_trace(fill(eval_record, fill_num)) for eval_record in eval_records
    ]
    

def write_log_to_csv(logs: List[EvalRecord], run_id: str, log_type: str, save_folder: str = '.'):
    log_file_path = os.path.join(save_folder,  f"{log_type}_{run_id}.csv")
    log_df = pd.DataFrame(data=map(flatten_dict, fill_traces(logs)))
    if not os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, index=False)
    else:
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)


class ActionAccu(TypedDict):
    pose0_position: Tuple[int, int]
    pose0_rotation: Tuple[int, int, int, int]
    pose1_position: Tuple[int, int]
    pose1_rotation: Tuple[int, int, int, int]

def per_attr_accu(oracle_action: Action, policy_action: Action) -> ActionAccu:
    for attr in ('pose0_position', 'pose0_rotation', 'pose1_position', 'pose1_rotation'):
        for i, (oracle, policy) in enumerate(zip(oracle_action[attr], policy_action[attr])):
            print(attr, i, int(oracle), int(policy))


def to_tensor_action(action: Action) -> Action:
    return {
        "pose0_position": torch.Tensor(action["pose0_position"].copy()),
        "pose1_position": torch.Tensor(action["pose1_position"].copy()),
        "pose0_rotation": torch.Tensor(action["pose0_rotation"].copy()),
        "pose1_rotation": torch.Tensor(action["pose1_rotation"].copy()),
    }

@torch.no_grad()
def eval_placement_generalization(
        model_path: str, 
        task: TaskName, 
        num_exp: int, 
        save_folder: str,
        seed: int
    ):
    run_config = RunConfig(
        partition="placement_generalization",
        task=task,
        ckpt=model_path,
        device="cuda"
    )
    model_name = os.path.basename(model_path).split('.')[0]
    env_config = EnvConfig(
        use_time_wrapper=True,
        use_reset_wrapper=True,
        make_config=MakeConfig(modalities=["segm", "rgb"],
        task_kwargs=PARTITION_TO_SPECS["test"][run_config.partition][run_config.task],
        seed=seed,
        render_prompt=False,
        display_debug_window=False,
        hide_arm_rgb=True)
    )
    if 'M' in model_path:
        prefix = ''
    else:
        prefix = 'module.'
    policy = create_policy_from_ckpt(
        run_config.ckpt, 
        run_config.device, 
        prefix=prefix
    )
    policy.eval()
    env = create_env(run_config, env_config)
    oracle_agent = env.task.oracle(env)
    for i in tqdm(range(seed, seed + num_exp)):
        env.seed(i)
        obs = env.reset()
        history = None
        eval_record: EvalRecord = {
            "seed": i,
            "failure": 0,
            "sucess": 0,
            "timeout": 0,
            "bad_action": 0,
            "task": run_config.task,
            "step_count": 0,
            "prompt": "",
            "action_trace": [],   
            "perf_trace": []
        }
        while True:
            action_start = time.time()
            policy_action, history = step(obs, env, policy, run_config.device, history)
            action_duration = time.time() - action_start
            oracle_action = oracle_step(obs, env, oracle_agent)
            if oracle_action is None:
                oracle_action = {
                    "pose0_position": [None, None],
                    "pose1_position": [None, None],
                    "pose0_rotation": [None, None, None, None],
                    "pose1_rotation": [None, None, None, None],
                }
                eval_record["action_trace"].append(
                    (
                        oracle_action, 
                        policy.discretize_action(to_tensor_action(policy_action))
                    )
                )
            else:
                eval_record["action_trace"].append(
                    (
                        policy.discretize_action(to_tensor_action(oracle_action)),
                        policy.discretize_action(to_tensor_action(policy_action))
                    )
                )
            if "obs_token_size" in history:
                eval_record["perf_trace"].append(
                    (
                        history["obs_token_size"] + history["prompt_token_size"], action_duration
                    )
                )
            else:
                eval_record["perf_trace"].append(
                    (
                        0, 0.0
                    )
                )
            obs, _, done, info = env.step(policy_action)
            eval_record["step_count"] += 1
            info: TaskInfo
            
            eval_record["prompt"] = info["prompt"]
            eval_record["sucess"] = int(info["success"])
            eval_record["failure"] = int(info["failure"])
            if "TimeLimit.truncated" in info:
                eval_record["timeout"] = int(info["TimeLimit.truncated"])
            else:
                eval_record["timeout"] = int(False)
            if done:
                break
        write_log_to_csv([eval_record], seed, f'eval_{model_name}', save_folder)
        if (eval_record["failure"] == 0 
            and eval_record["sucess"] == 0
            and eval_record["timeout"] == 0):
            eval_record["bad_action"] = 1
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="2M.ckpt")
    parser.add_argument("--task", type=str, default="visual_manipulation")
    parser.add_argument("--num_exp", type=int, default=100)
    parser.add_argument("--save", type=str, default='.')
    parser.add_argument("--seed", type=int, default=0)
    task_param = parser.parse_args()
    eval_placement_generalization(
        task_param.model_path, 
        task_param.task, 
        task_param.num_exp,
        task_param.save,
        task_param.seed
    )
    #main(task_param.model_path, task_param.task, task_param.num_exp)
