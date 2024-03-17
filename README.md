"# train_vima" 

# System Requirements

- Python 3.8
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
pip install -r requirements.txt
```

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

## Step 2 - run training script

Run the training logic and monitor the log csv file created by executing

```
python train_local.py
```

The instructions for DDP training will be added soon