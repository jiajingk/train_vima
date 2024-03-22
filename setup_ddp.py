import os
import json
from dotenv import dotenv_values
from remote_control.exec import direct_remote_execute, remote_execute_under_py_venv
from remote_control.util import send_small_file_to_server
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




if __name__ == "__main__":
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    files = [
        "train_local.py",
    ]
    sync_small_files(
        ip_lists, files, "train_vima"
    )
    