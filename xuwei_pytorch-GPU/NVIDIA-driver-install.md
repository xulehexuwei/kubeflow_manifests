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


- 4、安装推荐的版本驱动，这一步可以不安装，直接跳到安装cuda的时候一起安装

```shell
# 型号斟酌下，选取
sudo apt-get install nvidia-driver-515
```

- 5、使用以下命令检查 NVIDIA 驱动程序是否安装正确：

```shell
# Nvidia自带一个命令行工具可以查看显存的使用情况
sudo nvidia-smi
```

`至此，驱动安装完成。`

## 安装cuda

```shell
sudo sh cuda_11.2.2_460.32.03_linux.run
```


- 安装完配置环境变量，sudo vim ~/.bashrc

```shell

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
#export PATH=$PATH:/usr/local/cuda-11.2/bin
#export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.2

#export PATH=/usr/local/cuda-11.2/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH

export PATH=/usr/local/cuda-11.2/bin:/usr/local/cuda-11.2/nsight-compute-2020.3.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

- source ~/.bashrc


## 安装cudnn

根据系统版本[下载deb文件](https://developer.nvidia.com/rdp/cudnn-archive)

```shell
libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb
```

- 安装
```shell
#依次安装
sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb

#官方说法：To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v8 directory in the debian file.
#0. Copy the cuDNN sample to a writable path.

cp -r /usr/src/cudnn_samples_v8/ $HOME
#Go to the writable path.
cd  ~/cudnn_samples_v8/mnistCUDNN

#2. Compile the mnistCUDNN sample.
#编译文件。
sudo make clean 
sudo make # 编译有可能出错，执行 sudo apt-get install libfreeimage3 libfreeimage-dev 后再次 sudo make

# 3. Run the mnistCUDNN sample.
# 运行样例程序。
sudo ./mnistCUDNN

# 如果成功运行，会显示下列信息：pass

#查看cudnn版本
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## 安装nccl

- 两种方法

### 1- 通过github项目 nccl 安装

```shell
git clone https://github.com/NVIDIA/nccl.git

cd nccl

make src.build CUDA_HOME=/usr/local/cuda-11.2

```

then

```shell
# Install tools to create debian packages
sudo apt install build-essential devscripts debhelper fakeroot
# Build NCCL deb package
make pkg.debian.build
ls build/pkg/deb/

cd build/pkg/deb/
# 两个deb文件分别执行
sudo dpkg -i *.deb
```

### 2- 下载deb文件 安装
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
