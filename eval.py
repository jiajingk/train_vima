import torch
import numpy as np
from playground.typing import RunConfig, EnvConfig, TaskName
from playground.env import create_env, oracle_step, MakeConfig
from vima_bench import PARTITION_TO_SPECS
from playground.util.policy import create_policy_from_ckpt
from playground.api import step

np.set_printoptions(2)

@torch.no_grad()
def main(model_path: str, task: TaskName):
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
        render_prompt=False,
        display_debug_window=False,
        hide_arm_rgb=False)
    )
    policy = create_policy_from_ckpt(
        run_config.ckpt, 
        run_config.device, 
        prefix=''
    )
    policy.eval()
    env = create_env(run_config, env_config)
    oracle_agent = env.task.oracle(env)
    for i in range(100):
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


if __name__ == "__main__":
    main('2M.ckpt', 'visual_manipulation')