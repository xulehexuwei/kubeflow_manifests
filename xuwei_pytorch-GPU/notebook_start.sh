#!/bin/bash

# 基于镜像pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime 用 conda 制作的含有GPU 和 Pytorch环境的notebook
# conda install ipykernel
# conda install -c conda-forge nb_conda
# jupyter notebook --ip=0.0.0.0 --allow-root

docker stop gpu_notebook
docker rm gpu_notebook

docker run --name gpu_notebook -ti -d -p 8888:8888 \
--gpus all --ipc=host \
-v /home/autel/xuwei/notebook:/workspace \
pytorch_notebook_gpu:v1 \
jupyter notebook --ip=0.0.0.0 --allow-root
