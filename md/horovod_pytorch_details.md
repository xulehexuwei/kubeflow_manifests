## horovod-pytorch 数据并行分布式训练 case

一般来说，训练过程如下：

在训练的任何迭代中，给定一个随机的小批量，我们将该小批量中的样本分成\(k\)个部分，并将它们均匀地分在多个GPU上。
每个GPU根据分配给它的小批量子集计算模型参数的损失和梯度。
将(k)个GPU中每个GPU的局部梯度聚合以获得当前的小批量随机梯度。
聚合梯度被重新分配到每个GPU。
每个GPU使用这个小批量随机梯度来更新它维护的完整的模型参数集。


```python
import torch
import horovod.torch as hvd
 
hvd.init() # Horovod 使用 init 设置GPU 之间通信使用的后端和端口:
torch.cuda.set_device(hvd.local_rank()) # 声明训练会话的配置，一个 GPU 与一个进程绑定
 
# 使用 DistributedSampler 对数据集进行划分。如此前我们介绍的那样，它能帮助我们将每个 batch 划分成几个 partition，在当前进程中只需要获取和 rank 对应的那个 partition 进行训练：
train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
 
model = ...
model.cuda()

# 使用 DistributedOptimizer 包装优化器。它能帮助我们为不同 GPU 上求得的梯度进行 all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）。
# all reduce 后不同 GPU 中模型的梯度均为 all reduce 之前各 GPU 梯度的均值：
optimizer = optim.SGD(model.parameters())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# 使用 broadcast_parameters 包装模型参数，将模型参数从编号为 root_rank 的 GPU 复制到所有其他 GPU 中
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
 
for epoch in range(100):
    # 在训练的任何迭代中，给定一个小mini batch，指定GPU时底层代码会把小批量中的样本分成(k)个部分，并将它们均匀地分在多个GPU上。
    for batch_idx, (data, target) in enumerate(train_loader):
        # 下面的代码会把mini batch 按GPU的个数拆分到不同的GPU上，并行计算
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ...
        output = model(images)
        loss = criterion(output, target)
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 详解如下代码

```python
output = model(images)
loss = criterion(output, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- outputs = net(inputs) 即前向传播求出预测的值

- loss = criterion(outputs, labels) 这一步很明显，就是求loss

- optimizer.zero_grad() 意思是把梯度置零，也就是把loss关于weight的导数变成0.

- loss.backward() 即反向传播求梯度

- optimizer.step() 即更新所有参数，step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。

注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。

