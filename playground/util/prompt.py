import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from playground.typing import (
    TokenType, 
    Word,  
    Device, 
    PromptTokenEmbedding, 
    PromptMask, 
    ImageToken,
    VIMAEnvBase,
    VIMAPolicy,
    TaskName
)
from vima.utils import (
    any_concat,
    any_stack,
    any_to_datadict,
    any_to_torch_tensor,
    stack_sequence_fields,
)
from playground.tokenizer import get_placeholders, get_tokenizer
from tokenizers import Tokenizer
from einops import rearrange


def get_task_class(prompt: str) -> TaskName:
    """
    x 1)  'Put the {object}1 into the {object}2.' => 'visual_manipulation'
    x 2)  'Put the {texture}1 object in {scene} into the {texture}2 object.' => 'scene_understanding'
    x 3)  'Rotate the {object}1 {angles} degrees.' => 'rotate'
    x 4)  'Rearrange to this {scene}.' => 'rearrange'
    x 5)  'Rearrange objects to this setup {scene} and then restore.' => 'rearrange_then_restore'
    x 6)  '{demo object}1 is {novel adj} than {demo object}2. 
         Put the {adv} {novel adj} {object}1 into the {object}2.' => 'novel_adj'
    x 7)  'This is a {novel name}1 {object}1. This is a {novel name}2 {object}2. 
         Put {novel name}1 into a {novel name}2.' => 'novel_noun'
    x 8)  'This is a {novel name}1 {object}1. This is a {novel name}2 {object}2. 
         {demo object}1 is {adj} than {demo object}2. Put the {adv} {novel adj} 
         {novel name}1 into the {novel name}2.' => 'novel_adj_and_noun'
    x 9)  '"Twist" is defined as rotating object a specific angle. 
          For examples: From {before twist}i to {after twist}i. 
          Now twist all {texture} objects.' => 'twist'
    x 10) 'Follow this motion for {object}: {frame}1 ... {frame}i ... {frame}n.' => 'follow_motion'
    x 11) 'Stack objects in this order {frame}1...{frame}i...{frame}n.' => 'follow_order'
    x 12) 'Sweep {quantifier} {object} into {bounds} without exceeding {constraint}.' => 'sweep_without_exceeding'
    x 13) 'Sweep {quantifier} {object} into {bounds} without touching {constraint}.' => 'sweep_without_touching'
    x 14) 'Put all objects with the same texture as {object} into it.' => 'same_texture'
    x 15) 'Put all objects with the same profile as {object} into it.' => 'same_shape' / 'same_profile'
    x 16) 'First put {object}1 into {object}2 then 
         put the object that was previously 
         at its {direction} into the same {object}2.' => 'manipulate_old_neighbor'
    x 17) 'Put {object}1 into {object}2. 
         Finally restore it into its original container.' => 'pick_in_order_then_restore'
    """
    def contain(keywords: Tuple[str], prompt: str) -> bool:
        return bool( 
            all(
                element.lower() in prompt.lower() for element in keywords
            ) 
        )
    if contain(('put',), prompt):
        if contain(('finally'), prompt):
            return 'pick_in_order_then_restore'
        if contain(('first'), prompt):
            return 'manipulate_old_neighbor'
        if contain(('profile'), prompt):
            return 'same_profile'
        if contain(('all', 'texture'), prompt):
            return 'same_texture'
        if contain(('object'), prompt):
            return 'scene_understanding'
        if prompt.count('.') == 1:
            return 'visual_manipulation'
        if prompt.count('.') == 2 and contain(('than'), prompt):
            return 'novel_adj'
        if prompt.count('.') == 3 and contain(('this'), prompt):
            return 'novel_noun'
        if prompt.count('.') == 4 and contain(('this', 'than'), prompt):
            return 'novel_adj_and_noun'
    if contain(('sweep',), prompt):
        if contain(('exceeding'), prompt):
            return 'sweep_without_exceeding'
        if contain(('touching'), prompt):
            return 'sweep_without_touching'
    if contain(('stack',), prompt):
        return 'follow_order'
    if contain(('follow', 'motion'), prompt):
        return 'follow_motion'
    if contain(('twist',), prompt):
        return 'twist'
    if contain(('rotate',), prompt):
        return 'rotate'
    if contain(('rearrange',), prompt):
        if contain(('setup'), prompt):
            return 'rearrange_then_restore'
        else:
            return 'rearrange'
    return 'unknown'
    
        




def create_inital_prompt(env: VIMAEnvBase) -> Tuple[List[TokenType], Word, ImageToken]:
    prompt = env.prompt
    prompt_assets = env.prompt_assets
    return prepare_prompt(
        prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
    )

def create_prompt_token(
        prompt_token_type: List[TokenType], 
        word_batch: Word, 
        image_batch:  ImageToken,
        policy: VIMAPolicy, 
        device: Device
    ) -> Tuple[PromptTokenEmbedding, PromptMask]:
    word_batch = word_batch.to(device)
    image_batch = image_batch.to_torch_tensor(device=device)
    prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
        (prompt_token_type, word_batch, image_batch)
    )
    return (
        prompt_tokens, 
        prompt_masks
    )

def prepare_prompt(
        *, 
        prompt: str, 
        prompt_assets: Dict, 
        views: List[str], 
        tokenizer: Optional[Tokenizer] = None
    ) -> Tuple[List[TokenType], List[Word], List[ImageToken]]:
    views = sorted(views)
    if tokenizer is None:
        tokenizer = get_tokenizer()
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in get_placeholders()]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in get_placeholders():
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
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
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
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