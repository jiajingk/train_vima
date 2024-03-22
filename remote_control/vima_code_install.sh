pip install wheel==0.38.4
pip install setuptools==66
git clone https://github.com/jiajingk/train_vima.git
pip install git+https://github.com/vimalabs/VIMA
pip install gym==0.21.0
cd train_vima
wget https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt
pip install -r requirements.txt
