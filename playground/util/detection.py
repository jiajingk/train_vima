
import pathlib
import os
import torch
import numpy as np
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultPredictor
from typing import Optional

MRCNN: Optional[DefaultPredictor] = None

def get_mrcnn():   
    return DefaultPredictor(get_mrcnn_cfg())

def get_mrcnn_cfg():
    ckpt_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'mask_rcnn.pth')
    ckpt = torch.load(ckpt_path)
    cfg: CfgNode = get_cfg()
    ckpt_cfg = ckpt.pop("cfg")
    cfg.update(**ckpt_cfg)
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.WEIGHTS = ckpt_path
    return cfg



def get_segm(rgb_img: np.ndarray) -> np.ndarray:
    global MRCNN
    if MRCNN is None:
        MRCNN = get_mrcnn()
    outputs = MRCNN(
        rgb_img
    )
    segmentation_map = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    instances = outputs["instances"].to("cpu")
    obj_count = 1
    for i in range(len(instances)):
        if instances.scores[i] < 0.95:
            continue
        mask = instances.pred_masks[i].numpy()
        segmentation_map[mask] = obj_count
        obj_count += 1
    return segmentation_map