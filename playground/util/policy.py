from typing import Tuple, Optional
import os
import torch
from vima import VIMAPolicy
from playground.util.train import forward
from playground.typing import (
    NormalizedTraj,
    Device,
    VIMAPolicy,
    PredDist, 
    Action, 
    ForwardMetaData,
    TrainHistory,
    PolicyCfg
)
from vima.policy import VIMAPolicy
from typing import Tuple, List, Optional
import torch
import os


class VimaPolicyWraper(torch.nn.Module):
    def __init__(
        self,
        *,
        single_process_policy: VIMAPolicy,
        device: Device
        ):
        super().__init__()
        self.policy = single_process_policy
        self.device = device


    def forward(
            self, 
            data: List[NormalizedTraj]
        ) -> List[Tuple[PredDist, Action, ForwardMetaData]]:
        batch_preds = forward(
            data,
            self.device,
            self.policy
        )
        return batch_preds

def create_policy_from_ckpt(
        ckpt_path: str, 
        device: Device, 
        prefix: str = ''
    ) -> VIMAPolicy:
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    ckpt = torch.load(ckpt_path, map_location=device)
    policy_instance = VIMAPolicy(**ckpt["cfg"])
    policy_instance.load_state_dict(
        {k.replace(f"{prefix}policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )
    policy_instance.eval()
    return policy_instance.to(device)


def get_policy_and_cfg(
        ckpt_path: str, 
        device: Device, 
        prefix: str = '', 
        from_scratch: bool =True
    ) -> Tuple[VIMAPolicy, PolicyCfg, Optional[TrainHistory]]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt.pop('state_dict')
    if from_scratch:
        policy = VIMAPolicy(**ckpt["cfg"])
        policy.eval()
        return policy.to(device), ckpt["cfg"], None
    else:
        return create_policy_from_ckpt(ckpt_path, device, prefix), ckpt["cfg"], ckpt["history"]