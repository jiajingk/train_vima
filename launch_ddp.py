import os
import json
from dotenv import dotenv_values
from remote_control.exec import remote_execute_under_py_venv
from typing import List





if __name__ == "__main__":
    with open(dotenv_values('.env').get("AWS_IP_PATH")) as f:
        ip_lists = json.load(f)
    wandb_api_key = dotenv_values('.env').get("WANDB_API_KEY")
    commands = [
        "cd train_vima",
        "pip install wandb", 
        f"export WANDB_API_KEY={wandb_api_key}",
        "wandb login",
        ("python train_local.py", False)
    ]
    for remote_ip in ip_lists:
        remote_execute_under_py_venv(commands, {
            "pem_file_path": dotenv_values('.env').get("AWS_PEM_PATH"),
            "server_ip": remote_ip,
            "username": "ubuntu"
        }, 'venv')
    