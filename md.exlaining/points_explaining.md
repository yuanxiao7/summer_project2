## yolov3

yolov3有点话想讲



### 1. F. interpolate

F. interpolate ——**数组采样操作** torch.nn.functional. interpolate (input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None) 功能：利用插值方法，对输入的张量数组进行上下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。

### 2. bubbliiiing 数据集处理

在bubliiiing目录之下，找到VOCdevkit \ Main下找到数据集信息，voc数据集格式，train.txt一万多张用来训练，val.txt用来验证，用于生成路径文件。



### 3. 解析器

parser = argparse.ArgumentParser(description='test')  

这是一个方便你调参的函数，当你在终端运行程序时，想要修改某个参数或某个文件名，你就可以在对应的目录之下运行该文件，然后修改，如果是放在网络中的话，你可以在对应的函数中直接修改你的参数，类似于赋值，后面的直接调用就好。

```python
import argparse

parser1 = argparse.ArgumentParser(description='test')  # 解析器

parser1.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser1.add_argument('--seed', type=int, default=72, help='Random seed.')
parser1.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')

args = parser1.parse_args()  # 函数实例化
print(args.sparse)# 此处调用
```

### 4.  plot.show()与plot.pause（）



### 5. 数据增强之图片翻转

借鉴 https://zhuanlan.zhihu.com/p/104644392

### 6.datetime

```python
import datetime

print(datetime.datetime.now())  # 有微秒 

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # 只有年月日时分秒
```





