from typing import (
    List, 
    Dict, 
    Tuple, 
    TypedDict, 
    Literal, 
    Callable, 
    Optional,
    Union,
    Protocol,
)
from torch import IntTensor, BoolTensor, FloatTensor, Tensor
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from vima_bench import VIMAEnvBase
from vima_bench.tasks.components.encyclopedia import ProfilePedia
from vima import VIMAPolicy
from vima.nn.action_decoder.dists import MultiCategorical, Categorical
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

Device = Literal['cpu', 'cuda']
TokenType = int
Word = IntTensor  # int32

_ = VIMAEnvBase
_ = VIMAPolicy
_ = Categorical

class CroppedImg(TypedDict):
    front: Union[IntTensor, np.ndarray] # uint8
    top: Union[IntTensor, np.ndarray] # uint8

class BBox(TypedDict):
    front: Union[IntTensor, Tuple[int, int, int, int]] # int32
    top: Union[IntTensor, Tuple[int, int, int, int]]  # int32

class Mask(TypedDict):
    front: BoolTensor # bool
    top: BoolTensor # bool

class ImageToken(TypedDict):
    cropped_img: CroppedImg
    bbox: BBox
    mask: Mask

View = Literal['front', 'top']
Modality = Literal['segm', 'rgb']
EndEffector =  Union[int, np.ndarray]

class ActionBounds(TypedDict):
    low: np.ndarray # 2x1 float32
    high: np.ndarray # 2x1 float32

@dataclass
class SizeRange:
    low: Tuple[float, float, float]
    high: Tuple[float, float, float]



class ObjInfo(TypedDict):
    obj_name: str 
    obj_assets: str # path end with urdf
    obj_size_range: SizeRange 
    obj_from_template: bool
    obj_replace_fn: Optional[Callable]
    obj_pose_transform_fn: Optional[Callable]
    obj_alias: Optional[List[str]]
    obj_novel_name: Optional[str]
    obj_template_file: Optional[str] 
    obj_symmetry: Optional[float]
    obj_profile: ProfilePedia
    texture_name: str
    texture_color_value: Optional[Tuple[float, float, float]]
    texture_texture_asset: Optional[str] # path to jpg
    texture_alias: Optional[str]
    texture_novel_name: Optional[str]


class SegmData(TypedDict):
    top: IntTensor # T H W uint8
    front: IntTensor # T H W uint8

class ViewData(TypedDict):
    top: IntTensor # T C W H uint8
    front: IntTensor # T C W H uint8

class SingleStepViewData(TypedDict):
    top: IntTensor # C H W uint8
    front: IntTensor # C H W uint8

class SingleStepSegmData(TypedDict):
    top: IntTensor # H W uint8
    front: IntTensor # H W uint8

class ViewPatchData(TypedDict):
    top: IntTensor # B x T x N x C x W x H uint8
    front: IntTensor # B x T x N x C x W x H uint8

class BBoxData(TypedDict):
    top: IntTensor # B x T x N x 4 int32
    front: IntTensor # B x T x N x 4 int32

class ViewPatchMask(TypedDict):
    top: BoolTensor # B x T x N bool
    front: BoolTensor # B x T x N bool

class ObsData(TypedDict):
    segm: ViewData
    rgb: ViewData
    ee: EndEffector

class SgemData(TypedDict):
    segm: ViewData
    ee: EndEffector

class ObjectPatch(TypedDict):
    cropped_img: ViewPatchData
    mask: ViewPatchMask
    bbox: BBoxData
    

class PreparedObj(TypedDict):
    ee: Union[int, np.ndarray]
    objects: ObjectPatch


ObsTokenEmbedding = FloatTensor # B x T x (num_view x num_obj) x D
ObsMask = BoolTensor # B x T x (num_view x num_obj)
OracleAgent = namedtuple("OracleAgent", ["act"])

OracleActionBound = namedtuple("OracleActionBound", ["low", "high"])
PromptTokenEmbedding = FloatTensor # N x B x D
PromptMask = BoolTensor # B x N
ActionTokenEmbedding = FloatTensor # T x B x D


class OracleActionSpace(TypedDict):
    pose0_position: OracleActionBound
    pose0_rotation: OracleActionBound
    pose1_position: OracleActionBound
    pose1_rotation: OracleActionBound


@dataclass
class RunConfig:
    partition: str
    task: str
    ckpt: str
    device: str


@dataclass
class MakeConfig:
    modalities: List[str]
    task_kwargs: str
    seed: int
    render_prompt: bool
    display_debug_window: bool
    hide_arm_rgb: bool


@dataclass
class EnvConfig:
    use_time_wrapper: bool
    use_reset_wrapper: bool
    make_config: MakeConfig


class Action(TypedDict):
    pose0_position: Union[np.ndarray, Tensor]
    pose0_rotation: Union[np.ndarray, Tensor]
    pose1_position: Union[np.ndarray, Tensor]
    pose1_rotation: Union[np.ndarray, Tensor]


class BoundAction(TypedDict):
    """every field in the BoundAction is bounded by ActionBounds
    """
    pose0_position: Union[np.ndarray, Tensor]
    pose0_rotation: Union[np.ndarray, Tensor]
    pose1_position: Union[np.ndarray, Tensor]
    pose1_rotation: Union[np.ndarray, Tensor]

class BinAction(TypedDict):
    """every field in the BinAction is bounded by 0 ~ 1
    """
    pose0_position: Union[np.ndarray, Tensor]
    pose0_rotation: Union[np.ndarray, Tensor]
    pose1_position: Union[np.ndarray, Tensor]
    pose1_rotation: Union[np.ndarray, Tensor]


class ActionSpace(TypedDict):
    pose0_position: ActionBounds
    pose0_rotation: ActionBounds
    pose1_position: ActionBounds
    pose1_rotation: ActionBounds


class DiscreteAction(TypedDict):
    pose0_position: IntTensor
    pose0_rotation: IntTensor
    pose1_position: IntTensor
    pose1_rotation: IntTensor


class ContinuousAction(TypedDict):
    pose0_position: FloatTensor
    pose0_rotation: FloatTensor
    pose1_position: FloatTensor
    pose1_rotation: FloatTensor


class ObjInfo(TypedDict):
    obj_id: int
    obj_name: str
    obj_color: str


class CroppedImageList(TypedDict):
    cropped_img: List[np.ndarray]
    bbox: List[Tuple[int, int, int, int]]
    mask: List[np.ndarray]


class ViewPatchList(TypedDict):
    top: CroppedImageList
    front: CroppedImageList


class ObjList(TypedDict):
    ee: EndEffector
    objects: ViewPatchList


class PromptAsset(TypedDict, total=False):
    obj_info: List[ObjInfo]


class AssetViewData(TypedDict):
    segm: ViewData
    rgb: ViewData


class EnvHistory(TypedDict):
    obs_token_embeddings: List[ObsTokenEmbedding]
    obs_masks: List[ObsMask]
    action_tokens: List[ActionTokenEmbedding]
    prompt_token: Optional[PromptTokenEmbedding]
    prompt_mask: Optional[PromptMask]
    prompt: Optional[str]
    prompt_assets: Optional[Union[PromptAsset, Dict[str, AssetViewData]]]
    obses: List[ObsData]
    

DecodedAction = Union[ContinuousAction, DiscreteAction]
TaskInfo = TypedDict('TaskInfo', {'prompt': str, 'success': bool, 'failure': bool, 'TimeLimit.truncated': bool})


class EnvMetaInfo(TypedDict):
    end_effector_type: Literal['suction', 'spatula']
    n_objects: int
    difficulty: Literal['easy', 'medium', 'hard']
    views: List[View]
    modalities: List[Modality]
    seed: int
    action_bounds: ActionBounds
    robot_components: List[int]
    obj_id_to_info: Dict[int, ObjInfo]
    prompt: Optional[str]
    prompt_assets: Optional[Union[PromptAsset, Dict[str, AssetViewData]]]
    steps: int
    success: bool
    failure: bool


class TaskMetaData(TypedDict):
    n_steps_min: int
    n_steps_max: int
    n_steps_mean: float
    seed_min: int
    seed_max: int


class Traj(TypedDict):
    obs: ObsData
    action: Action
    meta: EnvMetaInfo


class CutTraj(TypedDict):
    obs: ObsData
    action: Optional[Action]
    meta: EnvMetaInfo


class NormalizedTraj(TypedDict):
    obs: ObsData
    action: Action
    meta: EnvMetaInfo


class Env(Protocol):
    meta_info: EnvMetaInfo
    prompt: str
    prompt_assets: Union[PromptAsset, Dict[str, AssetViewData]] 
    time_step: int
    current_task: Optional[Traj]


    @property
    def task(self) -> OracleAgent:
        ...

    def reset(self):
        ...

    def step(self, action: Action) -> Tuple[ObsData, int, bool, TaskInfo]:
        ...


class DataBatch(TypedDict):
    obs_action_tokens: Tensor
    obs_action_masks: Tensor
    prompt_tokens: Tensor
    prompt_masks: Tensor
    position_ids: Tensor
    prompt_position_ids: Tensor
    n_max_objs: List[int]
    target_actions: List[Action]
    token_lengths: List[int]
    is_rotation: List[bool]
    prompt: List[str]

class DecodeMeta(TypedDict):
    n_max_objs: int
    token_lengths: int
    target_actions: Action
    is_rotation: bool
    prompt: str

class PredDist(TypedDict):
    pose0_position: MultiCategorical
    pose0_rotation: MultiCategorical
    pose1_position: MultiCategorical
    pose1_rotation: MultiCategorical


class ForwardMetaData(TypedDict):
    is_rotation: bool
    action_length: int
    prompt: str


class StepMeasure(TypedDict):
    pose0_position: Tuple[Tensor, Tensor]
    pose0_rotation: Tuple[Tensor, Tensor, Tensor, Tensor]
    pose1_position: Tuple[Tensor, Tensor]
    pose1_rotation: Tuple[Tensor, Tensor, Tensor, Tensor]


class FlattenedStepMeasure(TypedDict):
    pose0_position_0: Union[float, Tensor]
    pose0_position_1: Union[float, Tensor]
    pose0_rotation_0: Union[float, Tensor]
    pose0_rotation_1: Union[float, Tensor]
    pose0_rotation_2: Union[float, Tensor]
    pose0_rotation_3: Union[float, Tensor]
    pose1_position_0: Union[float, Tensor]
    pose1_position_1: Union[float, Tensor]
    pose1_rotation_0: Union[float, Tensor]
    pose1_rotation_1: Union[float, Tensor]
    pose1_rotation_2: Union[float, Tensor]
    pose1_rotation_3: Union[float, Tensor]


TaskName = Literal[
    'follow_order',
    'manipulate_old_neighbor',
    'novel_adj',
    'novel_noun',
    'pick_in_order_then_restore',
    'rearrange',
    'rearrange_then_restore',
    'same_profile',
    'same_shape',
    'rotate',
    'scene_understanding',
    'simple_manipulation',
    'sweep_without_exceeding',
    'twist',
    'visual_manipulation',
    'unknown',
]


PredDistMapper = Callable[[PredDist, Tuple], Tensor]
ActionMapper = Callable[[Action, Tuple], Tensor]
Criterion = Callable[[Tensor, Tensor], Tensor]
StepEvaluator = Callable[[PredDist, Action, Criterion, PredDistMapper, ActionMapper, int], Tensor]


class PolicyCfg(TypedDict):
    embed_dim: int
    xf_n_layers: int
    sattn_n_heads: int
    xattn_n_heads: int


class ActionAxisWeight(TypedDict):
    pose0_position_0: float
    pose0_position_1: float
    pose1_position_0: float
    pose1_position_1: float
    pose0_rotation_0: float
    pose0_rotation_1: float
    pose0_rotation_2: float
    pose0_rotation_3: float
    pose1_rotation_0: float
    pose1_rotation_1: float
    pose1_rotation_2: float
    pose1_rotation_3: float


class TaskWeight(TypedDict):
    follow_order: float
    manipulate_old_neighbor: float
    novel_adj: float
    novel_noun: float
    pick_in_order_then_restore: float
    rearrange: float
    rearrange_then_restore: float
    same_profile: float
    rotate: float
    scene_understanding: float
    simple_manipulation: float
    sweep_without_exceeding: float
    follow_order: float
    twist: float
    visual_manipulation: float


ActionWeightMethod = Literal[
    'scale_to_same_order_of_magnitude',
    'default'
]


TaskWeightMethod = Literal[
    'scale_with_respect_to_the_max_batch_avg_loss',
    'scale_with_respect_to_the_min_batch_avg_loss',
    'default'
]


class DataFrameTrainLogSchema(TypedDict):
    unweigted_sample_loss__pose0_position_0: pd.Series
    unweigted_sample_loss__pose0_position_1: pd.Series
    unweigted_sample_loss__pose1_position_0: pd.Series
    unweigted_sample_loss__pose1_position_1: pd.Series
    unweigted_sample_loss__pose0_rotation_0: pd.Series
    unweigted_sample_loss__pose0_rotation_1: pd.Series
    unweigted_sample_loss__pose0_rotation_2: pd.Series
    unweigted_sample_loss__pose0_rotation_3: pd.Series
    unweigted_sample_loss__pose1_rotation_0: pd.Series
    unweigted_sample_loss__pose1_rotation_1: pd.Series
    unweigted_sample_loss__pose1_rotation_2: pd.Series
    unweigted_sample_loss__pose1_rotation_3: pd.Series
    axis_weight__pose0_position_0: pd.Series
    axis_weight__pose0_position_1: pd.Series
    axis_weight__pose1_position_0: pd.Series
    axis_weight__pose1_position_1: pd.Series
    axis_weight__pose0_rotation_0: pd.Series
    axis_weight__pose0_rotation_1: pd.Series
    axis_weight__pose0_rotation_2: pd.Series
    axis_weight__pose0_rotation_3: pd.Series
    axis_weight__pose1_rotation_0: pd.Series
    axis_weight__pose1_rotation_1: pd.Series
    axis_weight__pose1_rotation_2: pd.Series
    axis_weight__pose1_rotation_3: pd.Series
    task_weight__follow_order: pd.Series
    task_weight__manipulate_old_neighbor: pd.Series
    task_weight__novel_adj: pd.Series
    task_weight__novel_noun: pd.Series
    task_weight__pick_in_order_then_restore: pd.Series
    task_weight__rearrange: pd.Series
    task_weight__rearrange_then_restore: pd.Series
    task_weight__same_profile: pd.Series
    task_weight__rotate: pd.Series
    task_weight__scene_understanding: pd.Series
    task_weight__simple_manipulation: pd.Series
    task_weight__sweep_without_exceeding: pd.Series
    task_weight__twist: pd.Series
    task_weight__visual_manipulation: pd.Series
    task: pd.Series
    batch_id: pd.Series
    epoch_id: pd.Series


class CosAnnealingParam(TypedDict):
    warmup_end_at_iters: int
    learning_rate: float 
    flatten_end_at_iters: int
    lr_decay_end_at_iters: int
    min_lr: float


class TrainParam(TypedDict):
    model_size: str
    total_epoch: int
    local_batch_size: int
    distributed: bool


class OptimizerParam(TypedDict):
    clip_norm: float
    inital_lr: float
    weight_decay: float
    optimizer_name: str



class DDPParam(TypedDict):
    world_size: int
    master_ip: str
    master_port: str
    local_rank: int
    socket: str
    backend: Literal['nccl', 'gloo']


DatasetSource = Literal['s3://vima', 'local']


class DatasetParam(TypedDict):
    data_pct_usage: float
    validation_pct: float
    total_data_size_per_task: int
    tasks: List[TaskName]
    source: DatasetSource


class TrainHistory(TypedDict):
    optimizer_state_dict: Dict
    last_epoch: int
    wandb_config: Dict


class DistributedDataLoader(Protocol):
    sampler: DistributedSampler


LogTsUnit = str
LogTsValue = int
class TimedLog(TypedDict):
    measure: Dict[str, Union[int, float]]
    timestamp: Tuple[LogTsUnit, LogTsValue]

class SampleRecord(TypedDict):
    unweigted_sample_loss__pose0_position_0: float
    unweigted_sample_loss__pose0_position_1: float
    unweigted_sample_loss__pose1_position_0: float
    unweigted_sample_loss__pose1_position_1: float
    unweigted_sample_loss__pose0_rotation_0: float
    unweigted_sample_loss__pose0_rotation_1: float
    unweigted_sample_loss__pose0_rotation_2: float
    unweigted_sample_loss__pose0_rotation_3: float
    unweigted_sample_loss__pose1_rotation_0: float
    unweigted_sample_loss__pose1_rotation_1: float
    unweigted_sample_loss__pose1_rotation_2: float
    unweigted_sample_loss__pose1_rotation_3: float
    task: TaskName
    local_rank: int
    lr: float
    batch_id: int
    epoch_id: int


class WandbParam(TypedDict):
    project: str
    group: str
    job_type: str