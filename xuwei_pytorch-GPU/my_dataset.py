import torch
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, ):

        with open(data_path, "r") as f:
            self.data = f.read().splitlines()
            # 如果这里都爆内存的话，
            # 看起来只能使用文件指针，在getitem里边逐行读取了
            # 得到的data是 list[str]
        self.data_gen = self.get_data()

    def get_data(self):
        for idx, val in enumerate(self.data):
            yield idx, val

    def __len__(self):
        # 这里返回长度是用于tqdm进度条显示用的
        # 我这里乘以4是我之前预处理的时候看得到总量大概是文档数目的4倍
        # 你也可以设定一个很大的数字，当dataloader提取不到数据的时候就会停止
        return len(self.data)

    def __getitem__(self, idx):
        # 每次使用next函数返回生成器生成的一条数据，此处的idx用不到了
        # return x, y 一般返回这种格式训练集，标注集
        return next(self.data_gen)


if __name__ == '__main__':
    # DDP：从外部得到local_rank参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank

    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端


    dataset = MyDataset('test.txt')
    print(f"dataset length: {len(dataset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    # 假如有100条数据，2个GPU数据并行，那么每个GPU被分50条数据，这里batch_size=5，说明每个GPU下还会把50条数据拆分成每5条一份，这样每个进程中的min batch就是10（迭代10次）。
    # num_workers（创建多线程，提前加载未来会用到的batch数据）工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据。注意：这个数字必须是大于等于0的，负数估计会出错。
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=5, num_workers=4, sampler=train_sampler)

    print("success dataloader")

    for epoch in range(2):
        print(f"local_rank-{local_rank}-epoch-{epoch}")
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        dataloader.sampler.set_epoch(epoch)
        # 后面这部分，则与原来完全一致了。
        num = 0
        for idx, val in dataloader:
            # TODO 对于大数据集分布式训练，上面自定义的MyDataset可以返回的是数据文件的路径，在这一步处理的时候，拿文件路径去获取文件然后加载到内存中。
            print(f"local_rank-{local_rank}-epoch-{epoch}-num-{num}-idx-{idx}-val-{val}")
            num += 1

