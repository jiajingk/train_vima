import os
from vima_bench import PARTITION_TO_SPECS
count = 100
for task in PARTITION_TO_SPECS["test"]["placement_generalization"].keys():
    model = 'saved_model\\2024-04-03_leafy-lion-458_8.ckpt'
    command = f'python eval.py --model_path {model} --task {task} --num_exp {count}'
    os.system(command)