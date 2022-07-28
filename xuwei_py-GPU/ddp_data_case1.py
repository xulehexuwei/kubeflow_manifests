import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.distributed as dist


# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)

    print(f"total data: {len(my_trainset)}")

    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)

    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。假如有64条数据，2个GPU数据并行，那么每个GPU被分32条数据，
    # 这里batch_size=16，说明每个GPU下还会把32条数据拆分成每16条一份，这样每个进程中的min batch就是2（迭代2次）。
    # num_workers（创建多线程，提前加载未来会用到的batch数据）工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据。注意：这个数字必须是大于等于0的，负数估计会出错。
    trainLoader = torch.utils.data.DataLoader(my_trainset, batch_size=12500, sampler=train_sampler)
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

    # 准备训练数据
    trainLoader = get_dataset()

    import time
    time.sleep(30)

    iterator = tqdm(range(100))
    for epoch in iterator:
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果，每轮训练后打乱数据顺序，让下轮训练GPU拿到的训练数据不一样
        trainLoader.sampler.set_epoch(epoch)
        # 后面这部分，则与原来完全一致了。
        for batch_idx, (data, label) in enumerate(trainLoader):
            data, label = data.to(local_rank), label.to(local_rank)

            print(len(data), len(label))

            time.sleep(3)

################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 ddp_case1.py
