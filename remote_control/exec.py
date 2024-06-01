from remote_control.util import execute, SSHConfig
from typing import Optional, List, Iterator
from contextlib import contextmanager
from remote_control.typing import Command

SessionName = str




class TmuxSession:
    def __init__(self, session_name: str, config: SSHConfig):
        self.session_name = session_name
        self.config = config
        self.last_command_blocking = False
    
    def execute(self, command: Command):
        print(command)
        if isinstance(command, str):
            command_text, blocking = command, True
        else:
            command_text, blocking = command
        assert "'" not in command_text
        if blocking is True:
            execute(self.config, f"tmux send-keys -t {self.session_name} '{command_text}; tmux wait-for -S {self.session_name}_done' C-m")
            execute(self.config, f"tmux wait-for {self.session_name}_done")
        else:
            execute(self.config, f"tmux send-keys -t {self.session_name} '{command_text}' C-m")
        self.last_command_blocking = blocking


class PyVenv:
    def __init__(self, session: TmuxSession, env_path: str):
        self.session = session
        self.env_path = env_path
    
    def execute(self, command: Command):
        self.session.execute(command)


@contextmanager
def create_tmux_context(
        config: SSHConfig, 
        accept_duplicate: bool = True,
        exit_on_finish: bool = False,
        tmux_session: str = "command_execution"
    ) -> Iterator[TmuxSession]:
    _, err = execute(config, f"tmux new-session -d -s {tmux_session}")
    if not accept_duplicate:
        assert len(err) == 0, err
    execute(config, f"tmux send-keys -t {tmux_session} 'cd ~' C-m")
    yield TmuxSession(tmux_session, config)
    if exit_on_finish:
        execute(config, f"tmux kill-session -t {tmux_session}")


@contextmanager
def create_py_venv(
        session: TmuxSession, 
        venv_path: str
    ) -> Iterator[PyVenv]: 
    session.execute(f'cd {venv_path}')
    session.execute(f'source bin/activate')
    session.execute(f'cd ~')
    yield PyVenv(session, venv_path)
    if session.last_command_blocking is True:
        session.execute(f'deactivate')
    else:
        session.execute((f'deactivate', False))


def remote_execute_under_py_venv(
        commands: List[Command], 
        config: SSHConfig, 
        venv_path: str,
        accept_duplicate: bool = True,
        exit_on_finish: bool = False,
        env_name: str = 'command_execution'
    ):
    with create_tmux_context(
        config, accept_duplicate, exit_on_finish, env_name) as tmux_session:
        with create_py_venv(tmux_session, venv_path) as venv:
            for command in commands:
                print(f"pyenv executing: {command}")
                venv.execute(command)
    

def direct_remote_execute(
        commands: List[Command], 
        config: SSHConfig,
        accept_duplicate: bool = True,
        exit_on_finish: bool = False,
        env_name: str = 'command_execution'
    ):
    with create_tmux_context(
        config, accept_duplicate, exit_on_finish, env_name) as tmux_session:
        for command in commands:
            print(f"executing: {command}")
            tmux_session.execute(command)
            
