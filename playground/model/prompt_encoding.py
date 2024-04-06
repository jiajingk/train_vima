
from typing import List, Union, Literal, Dict, Tuple
from playground.typing import (
    TokenType, 
    ImageToken, 
    PromptAsset,
    View,
    SpecialToken,
    Word
)
from playground.util.prompt import prepare_prompt
from playground.tokenizer import get_tokenizer
from playground.util.obs import prepare_view_obj, prepare_view_obj_list
import numpy as np
import cv2
from einops import rearrange
from vima.utils import any_concat, any_stack, any_to_datadict, stack_sequence_fields, any_to_torch_tensor

def log_structure(data):
    import numpy as np
    import torch
    from vima.utils import DataDict
    if isinstance(data, dict) or isinstance(data, DataDict):
        return {key: log_structure(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [log_structure(element) for element in data]
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        return f"{type(data).__name__}: shape={data.shape} type={data.dtype}"
    else:
        return type(data).__name__


def encode_str():
    ...

def encode_img():
    ...

def get_special_tokens() -> List[SpecialToken]:
    return [
        "{base_obj}",
        "{base_obj_1}",
        "{base_obj_2}",
        "{dragged_obj}",
        "{dragged_obj_1}",
        "{dragged_obj_2}",
        "{dragged_obj_3}",
        "{dragged_obj_4}",
        "{dragged_obj_5}",
        "{swept_obj}",
        "{bounds}",
        "{constraint}",
        "{scene}",
        "{demo_blicker_obj_1}",
        "{demo_less_blicker_obj_1}",
        "{demo_blicker_obj_2}",
        "{demo_less_blicker_obj_2}",
        "{demo_blicker_obj_3}",
        "{demo_less_blicker_obj_3}",
        "{start_scene}",
        "{end_scene}",
        "{before_twist_1}",
        "{after_twist_1}",
        "{before_twist_2}",
        "{after_twist_2}",
        "{before_twist_3}",
        "{after_twist_3}",
        "{frame_0}",
        "{frame_1}",
        "{frame_2}",
        "{frame_3}",
        "{frame_4}",
        "{frame_5}",
        "{frame_6}",
        "{ring}",
        "{hanoi_stand}",
        "{start_scene_1}",
        "{end_scene_1}",
        "{start_scene_2}",
        "{end_scene_2}",
        "{start_scene_3}",
        "{end_scene_3}",
    ]

TokenizedPrompt = List[Union[int, SpecialToken]]

def tokenize(prompt: str) -> List[Union[int, SpecialToken]]:
    tokenizer = get_tokenizer()
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    return [
        prompt_token if prompt_token in get_special_tokens() else prompt_id
            for prompt_id, prompt_token in zip(prompt_ids, prompt_tokens)
    ]

def get_max_n_objs(raw_prompt: TokenizedPrompt, views: List[View]) -> Dict[View, int]:
    max_n_objs_prompt = {view: 0 for view in views}
    for token in raw_prompt:
        if isinstance(token, dict):
            for view in views:
                max_n_objs_prompt[view] = max(
                    max_n_objs_prompt[view], len(token["cropped_img"][view])
                )
    return max_n_objs_prompt


def to_obj_repr(asset: PromptAsset, views: List[View]) -> ImageToken:
    views = sorted(views)
    obj_info = asset["segm"]["obj_info"]
    placeholder_type = asset["placeholder_type"]
    if placeholder_type == "object":
        objects = [obj_info["obj_id"]]
    elif placeholder_type == "scene":
        objects = [each_info["obj_id"] for each_info in obj_info]
    obj_repr = {
        "cropped_img": {view: [] for view in views},
        "bbox": {view: [] for view in views},
    }
    for view in views:
        for obj_id in objects:
            result = prepare_view_obj(
                asset['rgb'][view],
                asset['segm'][view],
                obj_id,
            )
            if result is None:
                continue
            bboxes, cropped_imgs = result
            obj_repr["bbox"][view].append(bboxes)
            obj_repr["cropped_img"][view].append(cropped_imgs)
        obj_repr["bbox"][view] = np.asarray(obj_repr["bbox"][view])
        obj_repr["cropped_img"][view] = np.asarray(obj_repr["cropped_img"][view])
    return obj_repr

    
def padding_and_transform(
        prompts: List[TokenizedPrompt], 
        max_n_objs_prompt, 
        views
    ) -> Tuple[List[TokenType], List[Word], List[ImageToken]]:
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in prompts:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))
    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch