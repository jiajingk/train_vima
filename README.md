"# train_vima" 

# System Requirements

- Python 3.9
- Driver Version: 532.09       
- CUDA Version: 12.1

# Installation

Depending on your cuda version, pick the corresponding installation command from
https://pytorch.org/get-started/locally/

for cuda version 12.1, use

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

then install the rest of the dependencies

```
pip install git+https://github.com/vimalabs/VIMA
pip install wheel==0.38.4
pip install setuptools==66
pip install gym==0.21.0
pip install -r requirements.txt
```

## `.env` file setup

create `.env` file in the working directory with the following attributes

```
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
ABSOLUTE_PATH_OF_WORKING_DIR=""
```

for ddp training with wandb logging tools, some extra fields are needed in `.env` file 

```
AWS_PEM_PATH=""
AWS_IP_PATH=""
WANDB_API_KEY=""
DDP_MASTER_IP=""
DDP_MASTER_PORT=""
```

and you need to create a json file that contains the list of ip address of the ddp cluster
(for example `["123.123.123.123", "456.456.456.456"]`) and let `AWS_IP_PATH` point to this json file.
Also, all those remote machines should be ssh-able using `pem` file pointed by `AWS_PEM_PATH`

if you don't use AWS as remote source, you can leave the fields related to AWS blank


# Evaluation

As a start point, download [the 2M pretrained model](https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt) and put it in the working directory then run 

```
python eval.py
```

# Training

Next, if you want to fine tune or train the model from scratch, you can follow the steps below

## Step 1 - get the dataset

To get the dataset, you have 3 options

1. download [the dataset](https://huggingface.co/datasets/VIMA/VIMA-Data) to local directory (it's about 500GB)
2. generate the dataset through `scripts/data_generation/run.py`
3. write your own dataloader logic in `get_dataloader` function (located at `playground\dataloader.py`) to retrieve from remote source such as AWS S3 

The quickest way to get start is option 2. It can be done by run the following commands 
```
mkdir tasks
python scripts/data_generation/run.py
```
which will create the dataset in under the tasks folder, each task contains 4 trajectory

If you want to change the generation logic, you can change the configuration in `scripts/data_generation/conf.yaml`

After the data generation process finished, we can start training the model

## Step 2 - run training script

Run the training script and monitor the log csv file created during the execution

```
mkdir saved_model
python train_local.py
```

If we want to train from an existing model
```
mkdir parent_model
copy your_model.ckpt parent_model # for windows
python train_local.py
```

## Visualize logs

visualization of losses can be found in `visualize.ipynb`

## DDP Training

To deploy training uing DDP, one option is provision GPU machines on cloud, here we choose AWS, but
any ubuntu based GPU machines that support CUDA should do the work. The Image we use is `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 20.04) 20240318`. 

With the machines ready, we will use `run_ddp_command.py` and `train_ddp.py` to perform DDP training.
First, in `train_ddp.py`, based on the number of machine provisioned (`world_size`), changing the `local_batch_size` in `get_train_param` function to make sure `world_size * local_batch_size` is 128.

Then pick the function in `run_ddp_command` to do the work. The order is 
`install_ubuntu_dependencies` -> `install_python_dependencies` -> `sync_small_files` (`.env` and `train_ddp.py`) -> `launch`

A more user friendly setup will be added soon