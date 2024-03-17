from playground.typing import (
    PredDist,
    NormalizedTraj,
)
from playground.util.traj import (
    get_total_time_step
)



def log_step(
        traj: NormalizedTraj,
        pred_dist: PredDist, 
    ):
    for t_step in range(get_total_time_step(traj)):
        ...