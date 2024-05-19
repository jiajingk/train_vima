import os
import json
import time
from dotenv import dotenv_values
from remote_control.exec import direct_remote_execute, remote_execute_under_py_venv
from remote_control.util import (
    send_small_file_to_server,
    execute, 
    pull_latest_weight,
    pull_latest_file,
    pull_all_files, 
    send_large_file_to_server, 
)
from typing import List, Tuple
from playground.typing import InitalizeMode
from concurrent.futures import ThreadPoolExecutor


IPAddress = str

def clean_csv_logs(remote_ips: List[IPAddress]):
    def clean_csv_logs_task(remote_ip: IPAddress):
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        execute(
            config,
            'rm train_vima/*.csv'
        )        
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            clean_csv_logs_task, remote_ips
        )    

def clean_parent_weight(remote_ips: List[IPAddress]):
    def clean_parent_weight_task(remote_ip: IPAddress):
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        execute(
            config,
            'rm parent_model/*.ckpt'
        )
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            clean_parent_weight_task, remote_ips
        )    
    
        

def put_latest_weight(remote_ips: List[IPAddress], path):
    for remote_ip in remote_ips:
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        send_large_file_to_server(
            config,
            path,
            'parent_model'
        )

def get_all_csv_logs(remote_ips: List[IPAddress]) -> None:
    def get_all_csv_logs_tasks(remote_ip: remote_ips):
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        pull_all_files(config, "train_vima", "logs", "*.csv")
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            get_all_csv_logs_tasks, remote_ips
        )    
    

def get_latest_csv_logs(remote_ips: List[IPAddress]) -> None:
    def get_latest_csv_logs_task(remote_ip: remote_ips):
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        pull_latest_file(config, "train_vima", "logs", "eval_*.csv")
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            get_latest_csv_logs_task, remote_ips
        )    

def get_latest_weight(remote_ips: List[IPAddress]) -> str:
    master_ip = remote_ips[0]
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": master_ip,
        "username": "ubuntu"
    }
    name = pull_latest_weight(config, "saved_model", "saved_model")
    return name

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
    def sync_small_files_task(remote_ip: IPAddress):
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
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            sync_small_files_task, remote_ips
        )
        
            


def launch_eval(remote_ips: List[IPAddress]):
    wandb_api_key = dotenv_values('.env').get("WANDB_API_KEY")
    ddp_master_ip = dotenv_values('.env').get("DDP_MASTER_IP")
    ddp_master_port = dotenv_values('.env').get("DDP_MASTER_PORT")
    for i, remote_ip in enumerate(remote_ips):
        print(f"in machine: {remote_ip}")
        commands = [
            "cd train_vima",
            f"export WANDB_API_KEY={wandb_api_key}",
            "wandb login",
            (f"python eval_ddp.py --local_rank {i} --world_size 8 --master_ip {ddp_master_ip} --master_port {ddp_master_port}", False)
        ]
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv')



def launch_train(remote_ips: List[IPAddress], mode: InitalizeMode):
    wandb_api_key = dotenv_values('.env').get("WANDB_API_KEY")
    ddp_master_ip = dotenv_values('.env').get("DDP_MASTER_IP")
    ddp_master_port = dotenv_values('.env').get("DDP_MASTER_PORT")
    def launch_train_task(launch_config: Tuple[int, IPAddress]):
        i, remote_ip = launch_config
        commands = [
            "cd train_vima",
            f"export WANDB_API_KEY={wandb_api_key}",
            "wandb login",
            (f"python train_ddp.py --local_rank {i} "
             f"--world_size 8 --master_ip {ddp_master_ip} "
             f"--master_port {ddp_master_port} --train_mode {mode}"
             , False)
        ]
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv')

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            launch_train_task, zip(range(len(remote_ips)), remote_ips)
        )  

        

def sync_with_git(remote_ips: List[IPAddress]):
    commands = [
        "cd ~",
        "rm -rf train_vima",
        "git clone https://github.com/jiajingk/train_vima.git",
        "cd train_vima",
        "wget https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt",
        "wget https://huggingface.co/VIMA/VIMA/resolve/main/mask_rcnn.pth",
    ]
    def sync_with_git_task(remote_ip: IPAddress):
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv', accept_duplicate=False, exit_on_finish=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            sync_with_git_task, remote_ips
        )
    files = [
        "train_ddp.py",
        ".env",
    ]
    sync_small_files(
        remote_ips, files, "train_vima"
    )

def kill_all_tmux(remote_ips: List[IPAddress]):
    tmux_session = "command_execution" 
    def kill_tmux_task(remote_ip: IPAddress):
        print(remote_ip)
        config = {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }
        execute(config, f"tmux send-keys -t {tmux_session} C-c; tmux wait-for -S done")
        execute(config, f"tmux wait-for done")
        execute(config, f"tmux kill-session -t {tmux_session}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            kill_tmux_task, remote_ips
        )
        
        


def keep_training_alive(remote_ips: List[IPAddress], epoch: int):
    master_ip = remote_ips[0]
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": master_ip,
        "username": "ubuntu"
    }
    def is_live(ssh_config):
        tmux_session_name = 'command_execution'
        command = f"tmux list-panes -t {tmux_session_name} -F '#{{pane_current_command}}'"
        result, err = execute(ssh_config, command)
        if err:
            print(err)
            return False
        return 'python' in result.strip()
    count = 0
    while True:
        if is_live(config):
            count += 1
            print(f"{count}: is alive")
            time.sleep(300)
            continue
        print("detected training failure, restarting...")
        time.sleep(60)
        kill_all_tmux(remote_ips)
        clean_parent_weight(remote_ips)
        name = get_latest_weight(remote_ips)
        latest_epoch = int(name.split('.')[0].split('_')[-1])
        if latest_epoch >= epoch - 1:
            return
        put_latest_weight(remote_ips, f'saved_model\\{name}')
        launch_train(remote_ips, 'continous_from_ckpt')
        time.sleep(1800)


def fresh_train(remote_ips: List[IPAddress]):
    kill_all_tmux(remote_ips)
    clean_parent_weight(remote_ips)
    clean_csv_logs(remote_ips)
    files = [ "train_ddp.py", ".env" ]; sync_small_files(ip_lists, files, "train_vima")
    launch_train(ip_lists, 'random_init')



if __name__ == "__main__":
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    #sync_with_git(ip_lists)
    #get_latest_csv_logs(ip_lists)
    kill_all_tmux(ip_lists)
    #clean_parent_weight(ip_lists)
    #clean_csv_logs(ip_lists)
    #get_all_csv_logs(ip_lists)
    #name = get_latest_weight(ip_lists)
    #put_latest_weight(ip_lists, f'saved_model\\{name}')
    sync_with_git(ip_lists)
    fresh_train(ip_lists)
    #files = [ "train_ddp.py", ".env" ]; sync_small_files(ip_lists, files, "train_vima")
    #launch_train(ip_lists, 'ckpt_init')
    #keep_training_alive(ip_lists, 50)
    #get_all_csv_logs(ip_lists)
