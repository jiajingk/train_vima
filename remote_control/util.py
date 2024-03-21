import paramiko
import os
from tqdm import tqdm
from contextlib import contextmanager
from paramiko import SSHClient, SFTPClient
from typing import Iterator, TypedDict, List


class SSHConfig(TypedDict):
    server_ip: str
    username: str
    pem_file_path: str


@contextmanager
def get_ssh_client(config: SSHConfig) -> Iterator[SSHClient]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=config['server_ip'],
                       username=config['username'],
                       key_filename=config['pem_file_path'])
        yield client
    finally:
        client.close()


@contextmanager
def get_sftp_client(ssh_client: SSHClient) -> Iterator[SFTPClient]:
    try:
        sftp = ssh_client.open_sftp()
        yield sftp
    finally:
        sftp.close()


def execute(config: SSHConfig, command: str) -> str:
    with get_ssh_client(config) as client:
        _, stdout, stderr = client.exec_command(command)
        result = stdout.read().decode()
        err = stderr.read().decode()
        if err:
            print(f"Error: {err}")
        return result


def execute_multi_line(config: SSHConfig, commands: List[str]):
    execute(config, ';'.join(commands))


def receive_large_file_to_server(
        ssh_config: SSHConfig, 
        src_file_path: str, 
        dst_file_path: str
    ):
    with get_ssh_client(ssh_config) as ssh_client:
        with get_sftp_client(ssh_client) as sftp_client:
            file_size = sftp_client.lstat(src_file_path).st_size
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading from {ssh_config['server_ip']}") as pbar:
                def print_progress(transferred, to_be_transferred):
                    del to_be_transferred
                    pbar.update(transferred - pbar.n)
                sftp_client.get(src_file_path, dst_file_path, callback=print_progress)
            print(f"File successfully received {ssh_config['server_ip']}:{src_file_path}")


def pull_latest_weight(
        ssh_config: SSHConfig, 
        src_folder: str, 
        dst_folder: str
    ) -> str:
    command = f'find {src_folder} -name "*.ckpt" -ls -printf "%T@ %Tc %p\n" | sort -n'
    output = execute(ssh_config, command)
    last_location = output.split(' ')[-1].strip()
    model_name = last_location.split('/')[-1].strip()
    print(model_name)
    print(last_location)
    receive_large_file_to_server(ssh_config, last_location, f"{dst_folder}/{model_name}")
    return model_name


def send_small_file_to_server(
        ssh_config: SSHConfig, 
        local_file_path: str, 
        remote_file_path: str
    ):
    with get_ssh_client(ssh_config) as ssh_client:
        with get_sftp_client(ssh_client) as sftp_client:
            sftp_client.put(local_file_path, remote_file_path)


def send_large_file_to_server(
        ssh_config: SSHConfig,  
        local_file_path: str, 
        remote_directory: str
    ):
    remote_file_path = f"{remote_directory}/{local_file_path.split('/')[-1]}"
    with get_ssh_client(ssh_config) as ssh_client:
        with get_sftp_client(ssh_client) as sftp_client:
            file_size = os.path.getsize(local_file_path)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading to {ssh_config['server_ip']}") as pbar:
                def print_progress(transferred, to_be_transferred):
                    del to_be_transferred
                    pbar.update(transferred - pbar.n)
                sftp_client.put(local_file_path, remote_file_path, callback=print_progress)

