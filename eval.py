import torch
from playground.typing import RunConfig, TaskName
from playground.env import create_env, oracle_step
from playground.util.policy import create_policy_from_ckpt
from playground.api import step




@torch.no_grad()
def main(model_path: str, task: TaskName):
    run_config = RunConfig(
        partition="placement_generalization",
        task=task,
        ckpt=model_path,
        device="cuda"
    )
    policy = create_policy_from_ckpt(
        run_config.ckpt, 
        run_config.device, 
        prefix=''
    )
    policy.eval()
    env = create_env(run_config)
    import numpy as np
    np.set_printoptions(2)
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