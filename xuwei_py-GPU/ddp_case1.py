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

    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。假如有64条数据，2个GPU数据并行，那么每个GPU被分32条数据，
    # 这里batch_size=16，说明每个GPU下还会把32条数据拆分成每16条一份，这样每个进程中的min batch就是2（迭代2次）。
    # num_workers（创建多线程，提前加载未来会用到的batch数据）工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据。注意：这个数字必须是大于等于0的，负数估计会出错。
    trainLoader = torch.utils.data.DataLoader(my_trainset, batch_size=16, num_workers=2, sampler=train_sampler)
    return trainLoader

if __name__ == '__main__':
    ### 初始化我们的模型、数据、各种配置  ####
    # DDP：从外部得到local_rank参数
    # args.local_rank的参数: 通过torch.distributed.launch来启动训练，torch.distributed.launch 会给模型分配一个args.local_rank的参数，
    # 所以在训练代码中要解析这个参数，也可以通过torch.distributed.get_rank()获取进程id。
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    local_rank = args.local_rank

    # 初始化通信方式和端口，设定当前进程绑定的GPU号；nccl是 NVIDIA GPU 设备上最快、最推荐的后端
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    #### 进程内指定显卡
    # 目前很多场景下使用分布式都是默认一张卡对应一个进程，所以通常，我们会设置进程能够看到卡数： 下面例举3种操作的API，其本质都是控制进程的硬件使用。
    # 方式1：在进程内部设置可见的device
    # torch.cuda.set_device(args.local_rank)
    # 方式2：通过ddp里面的device_ids指定
    # ddp_model = DDP(model, device_ids=[rank])
    # 方式3：通过在进程内修改环境变量
    # os.environ['CUDA_VISIBLE_DEVICES'] = loac_rank
    #### 进程内指定显卡

    #### 分配模型（模型与GPU绑定）
    # 构造模型，如果模型已有训练参数，先加载 DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
    model = ToyModel().to(local_rank)
    ckpt_path = None # 已有模型参数文件(是否加载已有的模型参数)
    if dist.get_rank() == 0 and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))

    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    #### 分配模型（模型与GPU绑定）

    # 要在构造DDP model之后，才能用model初始化optimizer。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # 准备训练数据
    trainLoader = get_dataset()

    # 定义损失函数loss
    loss_func = nn.CrossEntropyLoss().to(local_rank)

    ### 网络训练  ###
    model.train()
    iterator = tqdm(range(100))
    for epoch in iterator:
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果，每轮训练后打乱数据顺序，让下轮训练GPU拿到的训练数据不一样
        trainLoader.sampler.set_epoch(epoch)
        # 后面这部分，则与原来完全一致了。
        for data, label in trainLoader:
            data, label = data.to(local_rank), label.to(local_rank)
            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_func(prediction, label)
            loss.backward()
            iterator.desc = "loss = %0.3f" % loss
            optimizer.step()

        # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。 因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
        # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), "%d.ckpt" % epoch)

################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp_case1.py