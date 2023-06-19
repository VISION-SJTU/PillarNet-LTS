FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc && \
echo "export PATH=/usr/local/cuda-11.3/bin:$PATH" >> ~/.bashrc && \
echo "export CUDA_HOME=/usr/local/cuda-11.3:$CUDA_HOME" >> ~/.bashrc

COPY ../docs/requirements.txt /opt
ENV FORCE_CUDA="1"
RUN pip --no-cache-dir install -r /opt/requirements.txt

