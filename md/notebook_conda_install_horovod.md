## 1.安装nccl

查看cuda版本cat /usr/local/cuda/version.txt，从https://developer.nvidia.com/nccl/nccl-download安装对应的包

```shell
sudo yum update
sudo yum install libnccl-2.5.6-1+cuda10.0 libnccl-devel-2.5.6-1+cuda10.0 libnccl-static-2.5.6-1+cuda10.0
```

## 2.安装openmpi

参考 https://blog.csdn.net/weixin_41010198/article/details/86294125

选择版本下载openmpi


```shell
./configure --prefix="/usr/local/openmpi"
make -j8
make install
export PATH="$PATH:/usr/local/openmpi/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/openmpi/lib/"
```

## 3. 安装horovod

首先安装好tensorflow

`conda install tensorflow-gpu=1.13.1`

执行安装命令，安装horovod

`HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 /data3/xingshuai/tools/miniconda3/bin/pip install --verbose --no-cache-dir horovod`

注意numpy版本要<1.17，执行miniconda3/bin/pip install numpy==1.16.4安装1.16版本

#### 3.1 horovod 也可以从源码安装

```shell
$ git clone --recursive https://github.com/horovod/horovod
$ cd horovod
$ python setup.py sdist
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir dist/horovod-0.16.4.tar.gz
```