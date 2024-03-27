import numpy as np
import cv2
import torch
import random
from typing import Optional, Tuple, List, Dict
from einops import rearrange
from vima.utils import (
    add_batch_dim,
    get_batch_size,
    any_slice,
    any_to_datadict,
    any_stack,
    any_transpose_first_two_axes,
    any_concat,
)
from playground.typing import (
    VIMAPolicy,
    EnvMetaInfo,
    Device,
    PreparedObj,
    ObsData,
    ObsTokenEmbedding,
    ObsMask,
    SingleStepViewData,
    ViewPatchData,
    EnvMetaInfo,
    ObjList,
    View,
    SegmData,
    ViewData,
)
from torch.distributions.categorical import Categorical

def pad_to_square(cropped_img: np.ndarray) -> np.ndarray:
    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
    if cropped_img.shape[1] > cropped_img.shape[2]:
        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
    else:
        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
    cropped_img = np.pad(
        cropped_img, pad_width, mode="constant", constant_values=0
    )
    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
    return cropped_img


def scale_to(cropped_img: np.ndarray, size: int) -> np.ndarray:
    cropped_img = rearrange(
        cropped_img, "c h w -> h w c"
    )
    cropped_img = np.asarray(cropped_img)
    cropped_img = cv2.resize(
        cropped_img,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    cropped_img = rearrange(
        cropped_img, 
        "h w c -> c h w"
    )
    return cropped_img

def prepare_view_random_noise(
        rgb_this_view: np.ndarray,
    ):
    img_height = rgb_this_view.shape[1]
    img_width = rgb_this_view.shape[2]
    max_bbox_height = img_height // 2
    max_bbox_width = img_width // 2
    h = np.random.randint(1, max_bbox_height + 1)
    w = np.random.randint(1, max_bbox_width + 1)
    ymin = np.random.randint(0, img_height - h + 1)
    xmin = np.random.randint(0, img_width - w + 1)
    ymax = ymin + h
    xmax = xmin + w
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
    if cropped_img.shape[1] != cropped_img.shape[2]:
        cropped_img = pad_to_square(cropped_img)
    cropped_img = scale_to(cropped_img, 32)
    return (int(x_center), int(y_center), int(h), int(w)), cropped_img


def prepare_view_obj(
        rgb_this_view: np.ndarray,
        segm_this_view: np.ndarray,
        obj_id: int
    ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    ys, xs = np.nonzero(segm_this_view == obj_id)
    if len(xs) < 2 or len(ys) < 2:
        return None
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    h, w = ymax - ymin, xmax - xmin
    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
    if cropped_img.shape[1] != cropped_img.shape[2]:
        cropped_img = pad_to_square(cropped_img)
    cropped_img = scale_to(cropped_img, 32)
    return (int(x_center), int(y_center), int(h), int(w)), cropped_img


def zero_pad_n_obj(datas: Tuple[np.ndarray], n_pad: int) -> Tuple[np.ndarray]:
    shapes = (data.shape for data in datas)
    padded_shapes = tuple(
        (n_pad,) + shape[1:] if len(shape) > 1 else n_pad for shape in shapes
    )
    return tuple(
        np.concatenate(
            [
                data,
                np.zeros(
                    padded_shape,
                    dtype=data.dtype,
                )
            ],
            axis=0,
        ) for data, padded_shape in zip(datas, padded_shapes)
    )

def cat(k: int, p: Dict[int, float]) -> int:
    assert set(p.keys()) == set(range(k)), "a valid distribution should acount all possible outcome"
    probs = [p[i] for i in range(k)]
    return (
        Categorical(torch.tensor(probs))
        .sample()
        .item()
    )


def prepare_view_obj_list(
        rgb_this_view: np.ndarray,
        segm_this_view: np.ndarray,
        obj_ids: List[int],
        apply_object_augmentation: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = []
    cropped_imgs = []
    n_pad = 0
    noise_ids = set()
    if (apply_object_augmentation and 
        (n_arg_objs := cat(2, {0: 0.95, 1: 0.05})) > 0):
        for i in range(n_arg_objs):
            random_index = np.random.randint(0, len(obj_ids))
            noise_id = max(obj_ids) + 1 + i
            noise_ids.add(noise_id)
            obj_ids.insert(random_index, noise_id)
    random.shuffle(obj_ids)
    for obj_id in obj_ids:
        if obj_id in noise_ids:
            view_obj = prepare_view_random_noise(
                rgb_this_view
            )
        else:
            view_obj = prepare_view_obj(
                rgb_this_view,
                segm_this_view,
                obj_id
            )
        if view_obj is None:
            n_pad += 1
            continue
        bbox, cropped_img = view_obj
        bboxes.append(bbox)        
        cropped_imgs.append(cropped_img)
    bboxes = np.asarray(bboxes)
    cropped_imgs = np.asarray(cropped_imgs)
    mask = np.ones(len(bboxes), dtype=bool)
    if n_pad > 0:
        bboxes, cropped_imgs, mask = zero_pad_n_obj(
            (bboxes, cropped_imgs, mask,), n_pad
        )
    return bboxes, cropped_imgs, mask
    

def prepare_obs(
    *,
    obs: ObsData,
    rgb_dict: Optional[SingleStepViewData] = None,
    meta: EnvMetaInfo,
    ) -> ViewPatchData:
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views: List[View] = sorted(rgb_dict.keys())
    assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = list(meta["obj_id_to_info"].keys())
    L_obs = get_batch_size(obs)
    obs_list: ObjList = {
        "ee": obs["ee"],
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }
    for l in range(L_obs):
        rgb_dict_this_step: ViewData = any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step: SegmData = any_slice(segm_dict, np.s_[l])
        for view in views:
            bboxes, cropped_imgs, mask = prepare_view_obj_list(
                rgb_dict_this_step[view],
                segm_dict_this_step[view],
                objects,
            )
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )
    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs = any_transpose_first_two_axes(obs)
    return obs


def get_obs_token_this_step(
        obs: ObsData, 
        meta_info: EnvMetaInfo, 
        policy: VIMAPolicy, 
        device: Device
    ) -> Tuple[ObsTokenEmbedding, ObsMask]:
    obs["ee"] = np.asarray(obs["ee"])
    obs = add_batch_dim(obs)
    obs: PreparedObj = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
        device=device
    )
    obs_token_embedding_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
    return obs_token_embedding_this_step, obs_mask_this_step


def obs_obj_padding(
        obs_token_embeddings: List[ObsTokenEmbedding],
        obs_masks: List[ObsMask],
        device: Device
    ) -> Tuple[List[ObsTokenEmbedding], List[ObsMask]]:
    max_objs = max(x.shape[0] for x in obs_token_embeddings)
    obs_tokens_this_env, obs_masks_this_env = [], []
    for idx in range(len(obs_token_embeddings)):
        obs_this_env_this_step = obs_token_embeddings[idx]
        obs_mask_this_env_this_step = obs_masks[idx]
        required_pad = max_objs - obs_this_env_this_step.shape[0]
        obs_tokens_this_env.append(
            any_concat(
                [
                    obs_this_env_this_step,
                    torch.zeros(
                        required_pad,
                        obs_this_env_this_step.shape[1],
                        device=device,
                        dtype=obs_this_env_this_step.dtype,
                    ),
                ],
                dim=0,
            )
        )
        obs_masks_this_env.append(
            any_concat(
                [
                    obs_mask_this_env_this_step,
                    torch.zeros(
                        required_pad,
                        device=device,
                        dtype=obs_mask_this_env_this_step.dtype,
                    ),
                ],
                dim=0,
            )
        )
    return obs_tokens_this_env, obs_masks_this_env


def prepare_obs_forward(
        padded_obs_token_embeddings: List[ObsTokenEmbedding],
        padded_obs_mask: List[ObsMask]
    ) -> Tuple[ObsTokenEmbedding, ObsMask]:
    obs_tokens_to_forward = [(any_stack(padded_obs_token_embeddings, dim=0))]
    obs_masks_to_forward = [(any_stack(padded_obs_mask, dim=0))]
    obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
    obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
    obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
    obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)
    return obs_tokens_to_forward, obs_masks_to_forward