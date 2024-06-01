import os
import json
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
    pull_latest_file
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

def run_eval(remote_ip: IPAddress, model_path: str, task: str, count: int, save_path: str):
    command = f'python eval.py --model_path {model_path} --task {task} --num_exp {count} --save {save_path}'
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


def remote_eval(remote_ip: IPAddress, model_path: str):
    count = 100
    save_path = '~/remote_eval/result'
    for task in PARTITION_TO_SPECS["test"]["placement_generalization"].keys():
        run_eval(
            remote_ip,
            model_path,
            task,
            count,
            save_path
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


def sync_model_to_eval(model_name: str, dst_ip: IPAddress):
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
    receive_large_file_from_server(
        src_config,
        f'saved_model/{model_name}.ckpt',
        f'tmp\\{model_name}.ckpt'
    )
    send_large_file_to_server(
        dst_config,
        f'tmp\\{model_name}.ckpt',
        'remote_eval/model'
    )
    os.remove(f'tmp\\{model_name}.ckpt')

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
    

if __name__ == "__main__":
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    unevaled_models = get_unevaled_models()
    payloads = list(zip(cycle(ip_lists), unevaled_models))
    with ThreadPoolExecutor(max_workers=len(ip_lists)) as executor:
        executor.map(
            remote_eval_task, payloads
        )
    