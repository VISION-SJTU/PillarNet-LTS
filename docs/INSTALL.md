## Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint)'s original document.

### Requirements

- Linux
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv) 

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.7.13 
- PyTorch: 1.12.1
- spconv-cu114
- CUDA: 11.3
- cudnn: 8.0

### Basic Installation 

```bash
# basic python libraries
conda create --name pillarnet python=3.7
conda activate pillarnet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

git clone https://github.com/WWDCRobot/PillarNet.git
cd PillarNet
pip install -r requirements.txt

# add PillarNet to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_PILLARNET"

# install cuda-ops package
bash setup.sh
```

### Advanced Installation 

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-11.3/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.3
export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Rotated NMS 
cd ROOT_DIR/det3d/ops/iou3d_nms
python setup.py build_ext --inplace
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data
