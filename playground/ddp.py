import os
import torch.distributed as dist
from playground.typing import (
    DDPParam,
)

def init_process(
        ddp_param: DDPParam
    ):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = ddp_param['master_ip']
    os.environ['MASTER_PORT'] = ddp_param['master_port']
    os.environ['NCCL_SOCKET_IFNAME'] = ddp_param['socket']
    os.environ['GLOO_SOCKET_IFNAME'] = ddp_param['socket']
    if ddp_param['local_rank'] == 0:
        print("master waiting for peer...")
    else:
        print(f"peer {ddp_param['local_rank']} looking for master...")
    dist.init_process_group(
        ddp_param['backend'], 
        init_method=f"tcp://{ddp_param['master_ip']}:{ddp_param['master_port']}", 
        rank=ddp_param['local_rank'], 
        world_size=ddp_param['world_size']
    )
    print("connected")
