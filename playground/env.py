from vima_bench import ALL_PARTITIONS, PARTITION_TO_SPECS, make
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
from dataclasses import asdict
from typing import Optional, Dict
from playground.typing import (
    OracleAgent,
    Action,
    RunConfig,
    EnvConfig,
    MakeConfig,
    VIMAEnvBase
)
import numpy as np

class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )


class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


def oracle_step(obs: Dict[str, Dict], env: VIMAEnvBase, oracle_agnet: OracleAgent) -> Optional[Action]:
    oracle_action: Action = oracle_agnet.act(obs)
    if oracle_action is None:
        print("WARNING: no oracle action, skip!")
        return None
    oracle_action = {
        k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
        for k, v in oracle_action.items()
    }
    return oracle_action


def create_env(run_cfg: RunConfig, env_config: Optional[EnvConfig] = None) -> VIMAEnvBase:
    assert run_cfg.partition in ALL_PARTITIONS
    assert run_cfg.task in PARTITION_TO_SPECS["test"][run_cfg.partition]
    if env_config is None:
        seed = 42
        env_config = EnvConfig(
            use_time_wrapper=True,
            use_reset_wrapper=True,
            make_config=MakeConfig(modalities=["segm", "rgb"],
            task_kwargs=PARTITION_TO_SPECS["test"][run_cfg.partition][run_cfg.task],
            seed=seed,
            render_prompt=True,
            display_debug_window=True,
            hide_arm_rgb=False)
        )
    
    if env_config.use_time_wrapper is True:
        time_wrapper = TimeLimitWrapper
    else:
        time_wrapper = lambda x, _: x
    if env_config.use_reset_wrapper is True:
        reset_wrapper = ResetFaultToleranceWrapper
    else:
        reset_wrapper = lambda x: x
    env = time_wrapper(
        reset_wrapper(
            make(
                run_cfg.task,
                **asdict(env_config.make_config)
            )
        ),
        bonus_steps=2,
    )
    return env