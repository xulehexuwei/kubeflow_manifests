## 1 前提条件

docker 已经可以使用GPU

[docker配置GPU](./GPU_docker_load.md)

### 1.1 MPI

[通信方式](./GPU-通信后端.md)

MPI 可用作 Gloo 的替代方案，用于协调 Horovod 中的进程之间的工作。使用 NCCL 时，两者的性能相似，但如果您进行 CPU 训练，则使用 MPI 有明显的性能优势。

## 2 容器单机多卡运行 horovod case

- 拉取镜像，此镜像很大（6G+)

[镜像链接](https://hub.docker.com/r/horovod/horovod/tags)

```shell
docker pull horovod/horovod:latest
```

- 进入容器，容器环境已经把很多依赖都装好了，比如 `nvidia-smi`、`torch`、`tensorflow`、`mpirun` 等
```shell
# --ipc=host 共享主机内存，单机多卡通信要用到
docker run --gpus all --rm -ti --ipc=host horovod/horovod:latest bash
```

- 执行`pip list`可以看到很多依赖包都装好了，这里要注意版本适配的问题，选择不同的 `horovod/horovod:latest` 镜像

![horovod-pip-list](../docs/images/horovod-pip-list.png)


### 2.1 运行一个case

- 容器中已经包含了很多case，选取其中一个，执行下面的命令运行：

```shell
docker run --gpus all -it --rm horovod/horovod:latest bash

# 进入/horovod/examples/pytorch目录，执行
horovodrun -np 2 -H localhost:2 python pytorch_mnist.py
```

- 由 pytorch 分布式训练脚本 [ddp_case1.py](./ddp_case1.py)修改后的[horovod脚本](./ddp_case1_horovod.py)
```shell
docker run -it --gpus all --rm -v /home/autel/xuwei/test_py:/horovod/examples/test_py horovod/horovod:latest bash
cd test_py

# 进入/horovod/examples/test_py目录，执行
horovodrun -np 4 -H localhost:4 python ddp_case1_horovod.py
# 或
mpirun --allow-run-as-root -n 4 --bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH python ddp_case1_horovod.py
```

上面的`horovodrun`命令等价 (单机多卡间无法通信，miss rank 或者 Connection reset by peer 通过启动docker 时加上 --ipc=host 共享主机内存解决)
```shell
mpirun --allow-run-as-root -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python pytorch_mnist.py
    
mpirun --allow-run-as-root -np 4  -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1  -mca pml ob1 -mca btl ^openib  python ddp_case1_horovod.py
```

![horovod-case](../docs/images/horovod-case.png)

从上面的图可以看出，单机多卡分布式训练已经启动了，本机调用 `watch -n 10 nvidia-smi` 命令，可以查看GPU的使用情况。


## 3- 在 horvod 容器中运行 pytorch ddp

镜像 `horovod/horovod:latest` 已经满足 `pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime` 的ddp环境，可以直接在 `horovod/horovod:latest`镜像启动的容器中进行ddp训练，即执行下面的命令：


- 一个 pytorch 分布式训练脚本 [ddp_case1.py](./ddp_case1.py)，运行下面的命令启动：

```shell
# /home/ubuntu/xuwei/test_py 是本机存放 ddp_case1.py 的目录，挂载到容器的 workspace 目录中
# PyTorch使用共享内存在进程之间共享数据，因此如果使用torch多处理（例如，对于多线程数据加载程序），容器运行的默认共享内存段大小是不够的，您应该使用--ipc=host或--shm-size命令行选项来增加共享内存大小以运行 docker。
# docker run --gpus all --rm -ti --ipc=host -v /home/autel/xuwei/test_py:/workspace  pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
docker run --gpus all --rm -ti --ipc=host -v /home/autel/xuwei/test_py:/workspace  horovod/horovod:latest

# 执行分布式训练命令，在启动容器中执行单机多卡分布式训练
# DDP: 使用torch.distributed.launch启动DDP模式; 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 ddp_case1.py
```
