import os
import json
import random
from functools import partial
from itertools import cycle
from typing import List, Tuple
from glob import glob
from dotenv import dotenv_values
from vima_bench import PARTITION_TO_SPECS
from concurrent.futures import ThreadPoolExecutor
from remote_control.util import (
    stream_all_files, 
    list_all_files, 
    receive_large_file_from_server, 
    send_large_file_to_server,
    pull_latest_file,
    execute
)
from remote_control.exec import remote_execute_under_py_venv, direct_remote_execute
IPAddress = str

def main():
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    master_ip = ip_lists[0]
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": master_ip,
        "username": "ubuntu"
    }
    for model_file_name in stream_all_files(config, 'saved_model', os.path.join('remote_eval', 'model'), '*.ckpt'):
        count = 100
        model = os.path.join('remote_eval', 'model', model_file_name)
        save_path = os.path.join('remote_eval', 'result')
        for task in PARTITION_TO_SPECS["test"]["placement_generalization"].keys():
            command = f'python eval.py --model_path {model} --task {task} --num_exp {count} --save {save_path}'
            os.system(command)
        os.remove(model)



def generate_random_numbers(n: int, k: int, min_val: int, max_val: int) -> List[int]:
    if (max_val - min_val + 1) < (n - 1) * k:
        raise ValueError("The range is too small to generate numbers with the given constraints.")
    
    numbers = []
    attempts = 0
    max_attempts = 1000  # limit the number of attempts to avoid infinite loop

    while len(numbers) < n and attempts < max_attempts:
        num = random.randint(min_val, max_val)
        if all(abs(num - existing_num) >= k for existing_num in numbers):
            numbers.append(num)
        attempts += 1

    if len(numbers) < n:
        raise ValueError("Couldn't generate the required number of random numbers with the given constraints.")

    return numbers


def setup_eval_folder(remote_ip: IPAddress):
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": remote_ip,
        "username": "ubuntu"
    }
    direct_remote_execute(
        [
            'mkdir -p remote_eval/model',
            'mkdir -p remote_eval/result'
        ],
        config, 
        accept_duplicate=False, 
        exit_on_finish=True, 
        env_name='eval'
    )

def run_eval(
        remote_ip: IPAddress, 
        model_path: str, 
        task: str, 
        count: int, 
        save_path: str,
        seed: int
    ):
    command = f'python eval.py --model_path {model_path} --task {task} --num_exp {count} --save {save_path} --seed {seed}'
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": remote_ip,
        "username": "ubuntu"
    }

    remote_execute_under_py_venv(
        [
            'cd train_vima',
            command
        ],
        config,
        'venv',
        accept_duplicate=False, exit_on_finish=True,
        env_name='eval_env'
    )


def remote_eval(remote_ip: IPAddress, model_path: str, seed: int = 0):
    count = 100
    save_path = '~/remote_eval/result'
    for task in PARTITION_TO_SPECS["test"]["placement_generalization"].keys():
        run_eval(
            remote_ip,
            model_path,
            task,
            count,
            save_path,
            seed
        )
    

def get_unevaled_models() -> List[str]:
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    master_ip = ip_lists[0]
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": master_ip,
        "username": "ubuntu"
    }
    all_models = list_all_files(config, 'saved_model', '*.ckpt')
    all_models = set(
        map(
            lambda x: x.split('/')[-1].split('.')[0], 
            all_models
        )
    )
    evaled_models = set(
        map(
            lambda x: '_'.join(os.path.basename(x).split('_')[1:4]), 
            glob(os.path.join('remote_eval', 'result', '*.csv'))
        )
    )
    models_to_eval = list(all_models - evaled_models)
    return models_to_eval


def kill_eval_tmux(remote_ip: IPAddress):
    tmux_session = "eval_env" 
    print(remote_ip)
    config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": remote_ip,
        "username": "ubuntu"
    }
    execute(config, f"tmux send-keys -t {tmux_session} C-c; tmux wait-for -S done")
    execute(config, f"tmux wait-for done")
    execute(config, f"tmux kill-session -t {tmux_session}")

def sync_model_to_eval(model_name: str, dst_ip: IPAddress, seed: int = 0):
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    master_ip = ip_lists[0]
    src_config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": master_ip,
        "username": "ubuntu"
    }
    dst_config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": dst_ip,
        "username": "ubuntu"
    }
    kill_eval_tmux(
        dst_ip
    )
    execute(
        dst_config,
        'rm remote_eval/model/*.ckpt'
    )
    execute(
        dst_config,
        'rm remote_eval/model/*.ckpt'
    )
    execute(
        dst_config,
        'rm remote_eval/result/*.csv'
    )
    receive_large_file_from_server(
        src_config,
        f'saved_model/{model_name}.ckpt',
        f'tmp\\{model_name}-{seed}.ckpt'
    )
    send_large_file_to_server(
        dst_config,
        f'tmp\\{model_name}-{seed}.ckpt',
        'remote_eval/model',
        remane=f'{model_name}.ckpt'
    )
    os.remove(f'tmp\\{model_name}-{seed}.ckpt')

def remote_eval_task(task_payload: Tuple[IPAddress, str]):
    remote_addr, model = task_payload
    sync_model_to_eval(model,  remote_addr)
    remote_eval(remote_addr, f'~/remote_eval/model/{model}.ckpt')
    src_config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": remote_addr,
        "username": "ubuntu"
    }
    pull_latest_file(
        src_config,
        'remote_eval/result',
        'remote_eval\\result',
        '*.csv'
    )


def remote_eval_seed(model_name: str, task_payload: Tuple[IPAddress, int]):
    remote_addr, seed = task_payload
    sync_model_to_eval(model_name,  remote_addr, seed)
    remote_eval(remote_addr, f'~/remote_eval/model/{model_name}.ckpt', seed)
    src_config = {
        "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
        "server_ip": remote_addr,
        "username": "ubuntu"
    }
    pull_latest_file(
        src_config,
        'remote_eval/result',
        'remote_eval\\result_per_seed',
        '*.csv'
    )


def eval_fix_seed():
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    unevaled_models = get_unevaled_models()
    payloads = list(zip(cycle(ip_lists), unevaled_models))
    with ThreadPoolExecutor(max_workers=len(ip_lists)) as executor:
        executor.map(
            remote_eval_task, payloads
        )

def eval_rand_seed(model_name: str):
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    task = partial(remote_eval_seed, model_name)
    payloads = list(zip(cycle(ip_lists), generate_random_numbers(16, 200, 0, 10000000)))
    with ThreadPoolExecutor(max_workers=min(len(ip_lists), len(payloads))) as executor:
        executor.map(
            task, payloads
        )


if __name__ == "__main__":
    #eval_rand_seed("2024-06-17_clear-moon-835_39")
    ...
    