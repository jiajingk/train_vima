"# train_vima" 

#### system requirement

Python 3.8
Driver Version: 532.09       
CUDA Version: 12.1

depends on your cuda version, pick the corresponding installation command from
https://pytorch.org/get-started/locally/

for cuda version 12.1, use

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

then install the rest of the dependencies

```
pip install -r requirements.txt
```

As a start point, download [the 2M pretrained model](https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt) and put it in the working directory then run 

```
python eval.py
```