import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# xw TODO
import horovod.torch as hvd


### 1. 基础模块 ###
# 假设我们的模型是这个
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
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

    print(f"total data {len(my_trainset)} hvd.size： {hvd.size()}")

    # 使用DistributedSampler，分布式采样器，可以封装自己的。用就完事儿！
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset, num_replicas=hvd.size(), rank=hvd.rank())

    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。假如有64条数据，2个GPU数据并行，那么每个GPU被分32条数据，
    # 这里batch_size=16，说明每个GPU下还会把32条数据拆分成每16条一份，这样每个进程中的min batch就是2（迭代2次）。
    trainLoader = torch.utils.data.DataLoader(my_trainset, batch_size=2000, sampler=train_sampler)
    return trainLoader

if __name__ == '__main__':

    # xw TODO
    hvd.init()
    # Horovod: pin GPU to local rank. 分配GPU到单个进程, 典型的设置是 1 个 GPU 一个进程，即设置 local rank。
    # if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())

    # 构造模型 xw TODO
    model = MyModel().to(hvd.local_rank())
    print(f"model cuda success")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # xw TODO 分布式优化器，包裹原来的优化器，进行all-reduce
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    print(f"DistributedOptimizer success")

    # xw TODO Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    print(f"broadcast cuda success")

    # 准备数据，分布式采样
    trainLoader = get_dataset()
    print(f"trainLoader success")
    # 假设我们的loss是这个
    loss_func = nn.CrossEntropyLoss().to(hvd.local_rank())

    ### 模型训练  ###
    print(f"start train")
    model.train()
    iterator = tqdm(range(100))
    for epoch in iterator:
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        trainLoader.sampler.set_epoch(epoch)
        # 后面这部分，则与原来完全一致了。
        for batch_idx, (data, label) in enumerate(trainLoader):
            data, label = data.to(hvd.local_rank()), label.to(hvd.local_rank())
            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_func(prediction, label)
            loss.backward()
            iterator.desc = "loss = %0.3f" % loss
            optimizer.step()

            # # 打印参数看看
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {}，hvd.rank: {} [{}/{}]\tLoss: {:.6f}'.format(
            #         epoch, hvd.rank(), len(data), len(trainLoader.sampler), loss.item()))

        # xw TODO 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
        if hvd.rank() == 0:
            # torch.save(net,path) —> 保存整个模型；
            # torch.save(net.state_dict(),path) --> 保存模型的参数。
            torch.save(model.state_dict(), "%d.ckpt" % epoch)
