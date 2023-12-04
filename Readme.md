# pytorch教程 李沐

[GitHub - ShusenTang/Dive-into-DL-PyTorch: 本项目将《动手学深度学习》(Dive into Deep Learning)原书中的MXNet实现改为PyTorch实现。](https://github.com/ShusenTang/Dive-into-DL-PyTorch/tree/master)

[跟着李沐【动手学深度学习】课程，大佬亲授全方面解读“花书”，带你从入门到精通（人工智能/深度学习/计算机视觉/图像处理）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1QP411j7jB/?spm_id_from=333.337.search-card.all.click&vd_source=a8ee18bc6643a102fd5b9ca976638dd0)

[《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/)



## pytorch构建模型

[深入浅出卷积神经网络及实现！ (qq.com)](https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg%3D%3D&idx=1&mid=2247499511&scene=21&sn=a420a254f767241e6b3c40e55b28a963#wechat_redirect)

```python
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic= False
torch.backends.cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
```

### pytorch常用网络

#### Linear全连接层

```python
nn.Linear(input_feature,out_feature,bias=True)
```



#### 2D卷积层

```python
nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups,bias=True,padding_mode='zeros')

##kernel_size,stride,padding 都可以是元组
## dilation 为在卷积核中插入的数量
```



#### 转置卷积 2D反卷积层

```python
nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=1,padding=0,out_padding=0,groups=1,bias=True,dilation=1,padding_mode='zeros')

##padding是输入填充，out_padding填充到输出
```



#### 最大值池化层 2D池化层

```python
nn.MaxPool2d(kernel_size,stride=None,padding=0,dilation=1)
```



#### 批量归一化层 2D归一化层

```python
nn.BatchNorm2d(num_features,eps,momentum,affine=True,track_running_stats=True)

affine=True 表示批量归一化的α，β是被学到的
track_running_stats=True 表示对数据的统计特征进行关注
```



### 创建模型的四种方法

假设创建卷积层–>Relu层–>池化层–>全连接层–>Relu层–>全连接层

```python
# 导入包import torch
import torch.nn.functional as F
from collections import OrderedDict
```

#### 自定义型(定义在init，前向过程在forward)

```python
class Net1(torch.nn.Module):
    def __init__(self):
      super(Net1, self).__init__()
      self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
      self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
      self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
      x = F.max_pool2d(F.relu(self.conv(x)), 2)
      x = x.view(x.size(0), -1)
      x = F.relu(self.dense1(x))
      x = self.dense2(x)
    return x
```

![图片](https://cdn.jsdelivr.net/gh/Jolene-hust/Jolene/img/202312041542231.png)

#### 序列集成型[利用nn.Squential(顺序执行的层函数)]

访问各层只能通过数字索引

```python
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
        torch.nn.Linear(32 * 3 * 3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out
```

![图片](https://cdn.jsdelivr.net/gh/Jolene-hust/Jolene/img/202312041543552.png)

#### 序列添加型[利用Squential类add_module顺序逐层添加]

给予各层的name属性

```python
class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv=torch.nn.Sequential()
        self.conv.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv.add_module("relu1",torch.nn.ReLU())
        self.conv.add_module("pool1",torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential()
        self.dense.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense.add_module("relu2",torch.nn.ReLU())
        self.dense.add_module("dense2",torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out
```

![图片](https://cdn.jsdelivr.net/gh/Jolene-hust/Jolene/img/202312041544650.png)



#### 序列集成字典型(OrderDict集成模型字典【‘name’:层函数】)

name为key

```python

lass Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = torch.nn.Sequential(
        OrderedDict(
        [
        ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
        ("relu1", torch.nn.ReLU()),
        ("pool", torch.nn.MaxPool2d(2))
        ]
        ))

        self.dense = torch.nn.Sequential(
        OrderedDict([
        ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
        ("relu2", torch.nn.ReLU()),
        ("dense2", torch.nn.Linear(128, 10))
        ])
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out
```

![图片](https://cdn.jsdelivr.net/gh/Jolene-hust/Jolene/img/202312041545890.png)

### 实战

- 构建网络模型：继承nn.Module函数的__init__ 函数，重定义前向传播函数forward
- 构造优化器
- 构造损失函数
- 训练 确定几个epoch【若运用数据增广，随机增广epoch次达到多样性】
- 对每个batch损失函数后向传播，优化器更新参数
-   optimizer.zero_grad() 清空梯度

-   loss.backward()

-   optimizer.step()









## U-NET教程

[Pytorch 深度学习实战教程（三）：UNet模型训练，深度解析！-CSDN博客](https://jackcui.blog.csdn.net/article/details/106349644?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-106349644-blog-105671859.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-106349644-blog-105671859.pc_relevant_antiscanv2&utm_relevant_index=1)



## 反向传播

[深度学习 | 反向传播详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/115571464)



## PVT代码详解

[Transformer主干网络——PVT_V2保姆级解析_pvtv2_只会git clone的程序员的博客-CSDN博客](https://blog.csdn.net/qq_37668436/article/details/122495068)