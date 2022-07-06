

## 1- 本机(物理机)要求

物理机要有显卡(GPU)

- 查看显卡信息：

```shell
lspci | grep -i vga
```


- 使用nvidia GPU可以：

```shell
lspci | grep -i nvidia
```

## 2- 本机显卡驱动安装（nvidia GPU 驱动）

首先使用 `nvidia-smi` 命令，测试是否已经安装好驱动，如果没该命令，按如下步骤安装：



安装 安装 cuda 和 cudnn

Nvidia自带一个命令行工具可以查看显存的使用情况：
```shell
nvidia-smi
```


## docker 容器使用 GPU

若 docker 版本 > 19.03 则不需要安装 nvidia-docker ，只需要安装 nvidia-container-tookit，步骤如下：

- 添加apt-get源
```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

- 安装`nvidia-container-toolkit`
```shell
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

- 验证安装是否成功
```shell
sudo docker run --rm --gpus all nvidia/cuda:10.0-base nvidia-smi
```
成功见下图
![成功](../docs/images/docker_gpu.png)

## docker pytorch (cuda-11.3)单机多卡容器启动训练

您还可以从 Docker Hub 拉取预先构建的 docker 映像并使用 docker v19.03+ 运行

- 注意：这里面pytorch版本和cuda版本一定要对应上，版本兼容问题是个大问题。

```shell
docker pull pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
```

- 上面的镜像启动的容器是可以在容器中使用GPU的


### 一个 pytorch 分布式训练脚本

```python
################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel().to(local_rank)

# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)


################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
```

将上面的脚本保存在 `ddp_case1.py` 文件中

### docker容器执行脚本进行单机多卡分布式训练

- `/home/ubuntu/xuwei/pytorch_ddp` 是本机存放 `ddp_case1.py` 的目录，挂载到容器的`workspace`目录中，运行下面的命令启动容器：

```shell
# PyTorch使用共享内存在进程之间共享数据，因此如果使用torch多处理（例如，对于多线程数据加载程序），容器运行的默认共享内存段大小是不够的，
# 您应该使用--ipc=host或--shm-size命令行选项来增加共享内存大小以运行nvidia-docker。
docker run --gpus all --rm -ti --ipc=host -v /home/ubuntu/xuwei/pytorch_ddp:/workspace  pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
```

- 注：`pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime`该镜像已经封装了conda、pip、`nvidia-smi`等环境，如下图所示：

![docker-nvidia-smi](../docs/images/docker-nvidia-smi.png)

- 执行分布式训练命令，启动容器中的单机多卡分布式训练

```shell
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp_case1.py
```

![pytorch_run](../docs/images/pytorch_run.png)

- 同时你可以在`本机`执行下面的命令，查看GPU使用情况：

```shell
# 周期性的输出显卡的使用情况，可以用watch指令实现，每隔10秒刷新一次使用情况：
watch -n 10 nvidia-smi
```

![pytorch_run_gpu](../docs/images/pytorch_run_gpu.png)
