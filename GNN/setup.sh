conda create -n ir python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate ir
# pip install torch_geometric
# pip3 install git+https://github.com/divelab/DIG.git
# pip install "networkx<3.0"
pip install rdkit-pypi==2020.9.5.2
# pip install "numpy>1.20.3"
# pip install pytorch-lightning==1.7.7
# pip install  torchmetrics==0.11.4
pip install jupyter gpustat scipy tqdm tensorboard 
pip install protobuf==4.25.3
ipython kernel install --name "ir" --user
echo done

conda create -n rdkit python=3.8 -y
conda activate rdkit
pip install rdkit jupyter
ipython kernel install --name "rdkit" --user