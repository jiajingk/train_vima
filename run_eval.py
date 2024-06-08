import os
from vima_bench import PARTITION_TO_SPECS
count = 100
for task in PARTITION_TO_SPECS["test"]["placement_generalization"].keys():
    model = '2M.ckpt'
    command = f'python eval.py --model_path {model} --task {task} --num_exp {count}'
    os.system(command)

# watch -n 5 "ls -lh eval_2024-05-16_graceful-vortex-638_25_2024-05-18.csv | awk '{print \$5}'"