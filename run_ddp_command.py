import os
import json
from dotenv import dotenv_values
from remote_control.exec import direct_remote_execute, remote_execute_under_py_venv
from remote_control.util import send_small_file_to_server, execute
from typing import List


IPAddress = str


def install_ubuntu_dependencies(remote_ips: List[IPAddress]):
    with open(os.path.join('remote_control', 'env_install.sh'), 'r') as f:
        commands = tuple(map(lambda x: x.strip(), f.readlines()))
    for command in commands:
        print(command)
    for remote_ip in remote_ips:
        direct_remote_execute(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, accept_duplicate=False, exit_on_finish=True
        )


def install_python_dependencies(remote_ips: List[IPAddress]):
    with open(os.path.join('remote_control', 'vima_code_install.sh'), 'r') as f:
        commands = tuple(map(lambda x: x.strip(), f.readlines()))
    for command in commands:
        print(command)
    for remote_ip in remote_ips:
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv', accept_duplicate=False, exit_on_finish=True)


def sync_small_files(
        remote_ips: List[IPAddress],
        file_paths: List[str],
        dst_folder: str
    ):
    for remote_ip in remote_ips:
        for file_path in file_paths: 
            send_small_file_to_server(
                {
                    "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
                    "server_ip": remote_ip,
                    "username": "ubuntu"
                },
                file_path,
                dst_folder
            )

def launch(remote_ips: List[IPAddress]):
    wandb_api_key = dotenv_values('.env').get("WANDB_API_KEY")
    ddp_master_ip = dotenv_values('.env').get("DDP_MASTER_IP")
    ddp_master_port = dotenv_values('.env').get("DDP_MASTER_PORT")
    for i, remote_ip in enumerate(remote_ips):
        print(f"in machine: {remote_ip}")
        commands = [
            "cd train_vima",
            f"export WANDB_API_KEY={wandb_api_key}",
            "wandb login",
            (f"python train_ddp.py --local_rank {i} --world_size 8 --master_ip {ddp_master_ip} --master_port {ddp_master_port}", False)
        ]
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv')

def sync_with_git(remote_ips: List[IPAddress]):
    commands = [
        "cd ~",
        "rm -rf train_vima",
        "git clone https://github.com/jiajingk/train_vima.git",
        "cd train_vima",
        "wget https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt",
    ]
    for remote_ip in remote_ips:
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv', accept_duplicate=False, exit_on_finish=True)
    files = [
        "train_ddp.py",
        ".env",
    ]
    sync_small_files(
        remote_ips, files, "train_vima"
    )

def kill_all_tmux(remote_ips: List[IPAddress]):
    tmux_session = "command_execution" 
    for remote_ip in remote_ips:
        print(remote_ip)
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        execute(config, f"tmux send-keys -t {tmux_session} C-c; tmux wait-for -S done")
        execute(config, f"tmux wait-for done")
        execute(config, f"tmux kill-session -t {tmux_session}")

        
if __name__ == "__main__":
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    ...
