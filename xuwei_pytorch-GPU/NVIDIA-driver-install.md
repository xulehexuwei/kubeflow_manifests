# NVIDIA A40 安装CUDA11.2+CUDNN8.1.1

[参考](https://blog.csdn.net/mtl1994/article/details/119039567)

## 参考

[安装教程](https://zhuanlan.zhihu.com/p/452075116)
[驱动下载](https://www.nvidia.com/download/index.aspx)


首先使用 `nvidia-smi` 命令，测试是否已经安装好驱动，如果没该命令，按如下步骤安装：

- 1、添加 nvidia repository

```shell
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

- 2、安装下载好的驱动文件

nvidia-driver-local-repo-ubuntu1804-460.106.00_1.0-1_amd64.deb

```shell
sudo dpkg -i nvidia-driver-local-repo-ubuntu1804-460.106.00_1.0-1_amd64.deb

sudo apt update
```

- 3、显示可用的驱动版本，按推荐的版本安装，推荐的应该就是第二步骤的版本，如下：

```shell
sudo ubuntu-drivers devices
```

![gpu_devices](../docs/images/gpu_devices.png)


## 安装nccl

NCCL (NVIDIA Colloctive Comunications Library)是英伟达的一款直接与GPU交互的库。
安装cupy前需要先安装该库。

1. 下载
官网下载地址：https://developer.nvidia.com/nccl/nccl-download
注意版本与你的cuda适配。

2.安装

获得.deb安装文件如：nccl-local-repo-ubuntu1804-2.8.4-cuda11.2_1.0-1_amd64.deb后

```shell
sudo dpkg -i nccl-local-repo-ubuntu1804-2.8.4-cuda11.2_1.0-1_amd64.deb  # 安装
# 如果提示缺少公共CUDA GPG秘钥
sudo apt-key add /var/nccl-repo-2.8.3-ga-cuda10.2/7fa2af80.pub

# 必不可少更新
sudo apt update
# 需指定版本，和上面的nccl一致
sudo apt install libnccl2=2.8.4-1+cuda11.2 libnccl-dev=2.8.4-1+cuda11.2
```
