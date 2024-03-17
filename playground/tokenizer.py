from __future__ import annotations

import os
from tokenizers import Tokenizer
from tokenizers import AddedToken
from typing import List

def get_placeholder_tokens() -> List[AddedToken]:
    _kwargs = {
        "single_word": True,
        "lstrip": False,
        "rstrip": False,
        "normalized": True,
    }
    return  [
        AddedToken("{base_obj}", **_kwargs),
        AddedToken("{base_obj_1}", **_kwargs),
        AddedToken("{base_obj_2}", **_kwargs),
        AddedToken("{dragged_obj}", **_kwargs),
        AddedToken("{dragged_obj_1}", **_kwargs),
        AddedToken("{dragged_obj_2}", **_kwargs),
        AddedToken("{dragged_obj_3}", **_kwargs),
        AddedToken("{dragged_obj_4}", **_kwargs),
        AddedToken("{dragged_obj_5}", **_kwargs),
        AddedToken("{swept_obj}", **_kwargs),
        AddedToken("{bounds}", **_kwargs),
        AddedToken("{constraint}", **_kwargs),
        AddedToken("{scene}", **_kwargs),
        AddedToken("{demo_blicker_obj_1}", **_kwargs),
        AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
        AddedToken("{demo_blicker_obj_2}", **_kwargs),
        AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
        AddedToken("{demo_blicker_obj_3}", **_kwargs),
        AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
        AddedToken("{start_scene}", **_kwargs),
        AddedToken("{end_scene}", **_kwargs),
        AddedToken("{before_twist_1}", **_kwargs),
        AddedToken("{after_twist_1}", **_kwargs),
        AddedToken("{before_twist_2}", **_kwargs),
        AddedToken("{after_twist_2}", **_kwargs),
        AddedToken("{before_twist_3}", **_kwargs),
        AddedToken("{after_twist_3}", **_kwargs),
        AddedToken("{frame_0}", **_kwargs),
        AddedToken("{frame_1}", **_kwargs),
        AddedToken("{frame_2}", **_kwargs),
        AddedToken("{frame_3}", **_kwargs),
        AddedToken("{frame_4}", **_kwargs),
        AddedToken("{frame_5}", **_kwargs),
        AddedToken("{frame_6}", **_kwargs),
        AddedToken("{ring}", **_kwargs),
        AddedToken("{hanoi_stand}", **_kwargs),
        AddedToken("{start_scene_1}", **_kwargs),
        AddedToken("{end_scene_1}", **_kwargs),
        AddedToken("{start_scene_2}", **_kwargs),
        AddedToken("{end_scene_2}", **_kwargs),
        AddedToken("{start_scene_3}", **_kwargs),
        AddedToken("{end_scene_3}", **_kwargs),
    ]

def get_placeholders() -> List[str]:
    return [
        token.content for token in get_placeholder_tokens()
    ]

def get_tokenizer() -> Tokenizer:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = Tokenizer.from_file('tokenizer.json')
    tokenizer.add_tokens(get_placeholder_tokens())
    return tokenizer