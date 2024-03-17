from typing import Tuple, Optional
import os
import torch
from vima import VIMAPolicy
from playground.typing import Device, PolicyCfg, TrainHistory


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