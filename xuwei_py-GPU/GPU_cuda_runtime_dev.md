
## cuda runtime 和 devel 区别（适应docker）

- CUDA：为“GPU通用计算”构建的运算平台。
- cudnn：为深度学习计算设计的软件库。
- CUDA Toolkit (nvidia)： CUDA完整的工具安装包，其中提供了 Nvidia 驱动程序、开发 CUDA 程序相关的开发工具包等可供安装的选项。包括 CUDA 程序的编译器、IDE、调试器等，CUDA 程序所对应的各式库文件以及它们的头文件。
- CUDA Toolkit (Pytorch)： CUDA不完整的工具安装包，其主要包含在使用 CUDA 相关的功能时所依赖的动态链接库。不会安装驱动程序。
- （NVCC 是CUDA的编译器，只是 CUDA Toolkit 中的一部分）
- 注：CUDA Toolkit 完整和不完整的区别：在安装了CUDA Toolkit (Pytorch)后，只要系统上存在与当前的 cudatoolkit 所兼容的 Nvidia 驱动，则已经编译好的 CUDA 相关的程序就可以直接运行，不需要重新进行编译过程。如需要为 Pytorch 框架添加 CUDA 相关的拓展时（Custom C++ and CUDA Extensions），需要对编写的 CUDA 相关的程序进行编译等操作，则需安装完整的 Nvidia 官方提供的 CUDA Toolkit。

- runtime的包，没有cuda的编译工具nvcc
- devel的包，是有cuda的nvcc包的

[Pytorch 使用不同版本的 cuda](https://www.cnblogs.com/yhjoker/p/10972795.html)

对于 Pytorch 之类的深度学习框架而言，其在大多数需要使用 GPU 的情况中只需要使用 CUDA 的动态链接库支持程序的运行( Pytorch 本身
与 CUDA 相关的部分是提前编译好的 )，就像常见的可执行程序一样，不需要重新进行编译过程，只需要其所依赖的动态链接库存在即可正常运行。故而，
Anaconda 在安装 Pytorch 等会使用到 CUDA 的框架时，会自动为用户安装 cudatoolkit，其主要包含应用程序在使用 CUDA 相关的功能时所依赖的动态链接库。
在安装了 cudatoolkit 后，只要系统上存在与当前的 cudatoolkit 所兼容的 Nvidia 驱动，则已经编译好的 CUDA 相关的程序就可以直接运行，而不需要安装完整的 Nvidia 官方提供的 CUDA Toolkit .
但对于一些特殊需求，如需要为 Pytorch 框架添加 CUDA 相关的拓展时( Custom C++ and CUDA Extensions )，需要对编写的 CUDA 相关的程序进行编译等操作，则需安装完整的 Nvidia 官方提供的 CUDA Toolkit.

- 所以，如果 Pytorch 框架添加 CUDA 相关的拓展时，要使用的docker镜像tag是devel版本，不需要时用runtime就可以（这个体积小很多）。

