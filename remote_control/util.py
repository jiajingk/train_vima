import paramiko
import os
from tqdm import tqdm
from contextlib import contextmanager
from paramiko import SSHClient, SFTPClient
from typing import Iterator, TypedDict, List, Tuple, Optional
import posixpath

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


def execute(config: SSHConfig, command: str) -> Tuple[str, str]:
    with get_ssh_client(config) as client:
        _, stdout, stderr = client.exec_command(command)
        result = stdout.read().decode()
        err = stderr.read().decode()
        if err:
            print(f"Error: {err}")
        return result, err


def execute_multi_line(config: SSHConfig, commands: List[str]):
    execute(config, ';'.join(commands))


def receive_large_file_from_server(
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


def pull_all_files(
        ssh_config: SSHConfig, 
        src_folder: str, 
        dst_folder: str,
        glob_pattern: str
    ) -> List[str]:
    command = f'find {src_folder} -name "{glob_pattern}" -ls -printf "%T@ %Tc %p\n" | sort -n'
    output, err = execute(ssh_config, command)
    assert len(err) == 0, err
    downloaded_files = []
    for line in output.strip().split('\n'):
        location = line.split(' ')[-1].strip()
        file_name = location.split('/')[-1].strip()
        if os.path.exists(f"{dst_folder}/{file_name}"):
            continue
        receive_large_file_from_server(ssh_config, location, f"{dst_folder}/{file_name}")
        downloaded_files.append(file_name)
    return downloaded_files

def stream_all_files(
        ssh_config: SSHConfig, 
        src_folder: str, 
        dst_folder: str,
        glob_pattern: str
    ) -> Iterator[str]:
    command = f'find {src_folder} -name "{glob_pattern}" -ls -printf "%T@ %Tc %p\n" | sort -n'
    output, err = execute(ssh_config, command)
    assert len(err) == 0, err
    for line in output.strip().split('\n'):
        location = line.split(' ')[-1].strip()
        file_name = location.split('/')[-1].strip()
        if os.path.exists(f"{dst_folder}/{file_name}"):
            continue
        receive_large_file_from_server(ssh_config, location, f"{dst_folder}/{file_name}")
        yield file_name


def list_all_files(
        ssh_config: SSHConfig, 
        src_folder: str, 
        glob_pattern: str
    ) -> List[str]:
    command = f'find {src_folder} -name "{glob_pattern}" -ls -printf "%T@ %Tc %p\n" | sort -n'
    output, err = execute(ssh_config, command)
    assert len(err) == 0, err
    locations = []
    for line in output.strip().split('\n'):
        location = line.split(' ')[-1].strip()
        locations.append(location)
    return locations

def pull_latest_file(
        ssh_config: SSHConfig, 
        src_folder: str, 
        dst_folder: str,
        glob_pattern: str
    ) -> str:
    command = f'find {src_folder} -name "{glob_pattern}" -ls -printf "%T@ %Tc %p\n" | sort -n'
    output, err = execute(ssh_config, command)
    assert len(err) == 0, err
    print(output)
    last_line = output.strip().split('\n')[-1]
    last_location = last_line.split(' ')[-1].strip()
    file_name = last_location.split('/')[-1].strip()
    print(last_location)
    receive_large_file_from_server(ssh_config, last_location, f"{dst_folder}/{file_name}")
    return file_name


def pull_latest_weight(
        ssh_config: SSHConfig, 
        src_folder: str, 
        dst_folder: str,
    ) -> str:
    command = f'find {src_folder} -name "*.ckpt" -ls -printf "%T@ %Tc %p\n" | sort -n'
    output, err = execute(ssh_config, command)
    assert len(err) == 0, err
    last_location = output.split(' ')[-1].strip()
    model_name = last_location.split('/')[-1].strip()
    print(model_name)
    print(last_location)
    receive_large_file_from_server(ssh_config, last_location, f"{dst_folder}/{model_name}")
    return model_name


def send_small_file_to_server(
        ssh_config: SSHConfig, 
        local_file_path: str, 
        remote_directory: str,
        remane: Optional[str] = None
    ):
    if remane is None:
        remote_file_name = os.path.basename(local_file_path)
    else:
        remote_file_name = remane
    with get_ssh_client(ssh_config) as ssh_client:
        with get_sftp_client(ssh_client) as sftp_client:
            sftp_client.put(local_file_path, posixpath.join(remote_directory, remote_file_name))


def send_large_file_to_server(
        ssh_config: SSHConfig,  
        local_file_path: str, 
        remote_directory: str,
        remane: Optional[str] = None
    ):
    if remane is None:
        remote_file_name = os.path.basename(local_file_path)
    else:
        remote_file_name = remane
    
    with get_ssh_client(ssh_config) as ssh_client:
        with get_sftp_client(ssh_client) as sftp_client:
            file_size = os.path.getsize(local_file_path)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading to {ssh_config['server_ip']}") as pbar:
                def print_progress(transferred, to_be_transferred):
                    del to_be_transferred
                    pbar.update(transferred - pbar.n)
                sftp_client.put(local_file_path, posixpath.join(remote_directory, remote_file_name), callback=print_progress)

