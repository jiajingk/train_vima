from typing import TypedDict, List, Tuple, Dict, Union
from playground.typing import (
    TimedLog, SampleRecord, DataFrameTrainLogSchema
)
import pandas as pd


def flatten_nested(d: Union[Dict, List], parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_dict(d: Dict[str, Union[str, int, float]], parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def measure_avg_accu(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_accu_cols = [
        str(col) for col in df.columns if 'sample_accu' in col
    ]
    measure = {
        "avg_sample_accu": df[unweighted_sample_accu_cols].sum(axis=1).mean()
    }
    
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_avg_position_accu(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_accu_cols = [
        str(col) for col in df.columns if 'sample_accu' in col
    ]
    unweighted_sample_accu_position_cols = [
        str(col) for col in unweighted_sample_accu_cols if 'position' in col
    ]
    measure = {
        "avg_sample_position_accu": df[unweighted_sample_accu_position_cols].mean(axis=1).mean()
    }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_avg_rotation_accu(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_accu_cols = [
        str(col) for col in df.columns if 'sample_accu' in col
    ]
    unweighted_sample_accu_rotation_cols = [
        str(col) for col in unweighted_sample_accu_cols if 'rotation' in col
    ]
    df = df.loc[ (df['task'] == 'twist') | (df['task'] == 'rotate') ]
    if len(df) == 0:
        measure = {
            "avg_sample_rotation_accu": 0
        }
    else:
        measure = {
            "avg_sample_rotation_accu": df[unweighted_sample_accu_rotation_cols].mean(axis=1).mean()
        }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_sample_accu_per_attribute(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'sample_accu' in col
    ]
    measure = df[unweighted_sample_loss_cols].mean().to_dict()
    
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }


def measure_sample_accu_per_task(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'sample_accu' in col
    ]
    measure = df[unweighted_sample_loss_cols + ["task"]].groupby("task").sum().mean(axis=1).to_dict()
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }


def measure_unweighted_loss_per_attribute(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'unweigted_sample_loss' in col
    ]
    position_cols = [
        str(col) for col in unweighted_sample_loss_cols if 'position' in col
    ]
    rotation_cols = [
        str(col) for col in unweighted_sample_loss_cols if 'rotation' in col
    ]
    position_measure = df[position_cols].mean().to_dict()
    rotation_df = df.loc[ (df['task'] == 'twist') | (df['task'] == 'rotate') ]
    if len(rotation_df) == 0:
        rotation_measure = {
           key:0 for key in rotation_cols
        }
    else:
        rotation_measure = df[rotation_cols].mean().to_dict()
    
    measure = {
        **position_measure,
        **rotation_measure
    }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_unweighted_position_loss(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'unweigted_sample_loss' in col
    ]
    unweighted_sample_position_loss_cols = [
        str(col) for col in unweighted_sample_loss_cols if 'position' in col
    ]
    measure = {
        "unweighted_sample_position_loss": df[unweighted_sample_position_loss_cols].mean(axis=1).mean()
    }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_unweighted_rotation_loss(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'unweigted_sample_loss' in col
    ]
    unweighted_sample_rotation_rotate_cols = [
        str(col) for col in unweighted_sample_loss_cols if 'rotation' in col
    ]
    df = df[(df.loc[df['task'] == 'twist']) | (df.loc[df['task'] == 'rotate'])]
    if len(df) == 0:
        measure = {
            "unweighted_sample_rotation_loss": 0
        }
    else:
        measure = {
            "unweighted_sample_rotation_loss": df[unweighted_sample_rotation_rotate_cols].mean(axis=1).mean()
        }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_unweighted_loss_per_task(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'unweigted_sample_loss' in col
    ]
    measure = df[unweighted_sample_loss_cols + ["task"]].groupby("task").mean().mean(axis=1).to_dict()
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_unweighted_loss_per_task_per_attribute(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'unweigted_sample_loss' in col
    ]
    measure = df[unweighted_sample_loss_cols + ["task"]].groupby("task").mean().to_dict()
    return {
        "measure": {prefix + key: value for key, value in flatten_dict(measure).items()},
        "timestamp": time_stamp
    }


def measure_unweighted_loss(
        logs: DataFrameTrainLogSchema, 
        epoch_id: int,
    ) -> float:
    log_df = pd.DataFrame(data=logs)
    new_columns = {
        col: col.replace('validation__', '') 
            for col in log_df.columns if col.startswith('validation__')
    }
    log_df = log_df.rename(columns=new_columns)
    unweighted_sample_cols = [
        col for col in log_df.columns 
            if 'unweigted_sample_loss' in col
    ]
    desired_cols = log_df.columns.isin(unweighted_sample_cols)
    epoch_loss = ( 
        log_df
        .loc[log_df['epoch_id'] == epoch_id]
        .iloc[:, desired_cols]
        .sum(axis=1)
        .mean()
    )
    return float(epoch_loss)

def measure_avg_lr(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    measure = {
        "lr": records[0]["lr"]
    }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

def measure_avg_unweighted_loss(
        records: List[SampleRecord],
        time_stamp: Tuple[str, int],
        prefix: str = '',
    ) -> TimedLog:
    df = pd.DataFrame(data = records)
    unweighted_sample_loss_cols = [
        str(col) for col in df.columns if 'unweigted_sample_loss' in col
    ]
    measure = {
        "avg_unweigted_sample_loss": df[unweighted_sample_loss_cols].sum(axis=1).mean()
    }
    return {
        "measure": {prefix + key: value for key, value in measure.items()},
        "timestamp": time_stamp
    }

