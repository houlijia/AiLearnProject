# 导入 matplotlib 配置模块，用于在导入 pyplot 之前设置后端
import matplotlib
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络层模块 (如卷积层、全连接层)
import torch.nn as nn
# 导入 PyTorch 神经网络函数模块 (如激活函数 ReLU, max_pool)
import torch.nn.functional as F
# 导入 PyTorch 优化器模块 (如 SGD, Adam)
import torch.optim as optim
# 导入 torchvision 库，包含常用数据集、模型架构和图像变换工具
import torchvision
# 导入图像变换工具包，并重命名为 transforms (方便后续调用)
import torchvision.transforms as transforms
# 从 torch.utils.data 导入数据加载器 DataLoader 和张量数据集 TensorDataset
# 注意：原代码拼写为 torch.utils.data，这里保持正确引用
from torch.utils.data import DataLoader, TensorDataset

# 【重要修正】设置 Matplotlib 的后端为 'TkAgg' (注意大小写，原代码 TKAgg 会报错)
# 这必须在 import matplotlib.pyplot 之前执行，否则设置无效
# TkAgg 是基于 Tkinter 的交互式后端，适合 macOS 和 Linux
matplotlib.use('TkAgg')

# 导入 matplotlib 的绘图接口
import matplotlib.pyplot as plt
# 导入 numpy 数值计算库
import numpy as np

# 开启 Matplotlib 的交互模式 (Interactive Mode)
# 开启后，plt.show() 不会阻塞程序运行，图片会在后台窗口显示，代码继续向下执行
plt.ion()


# 定义一个用于显示图像的辅助函数
def imshow(img):
    # 反归一化图像像素值
    # 训练时我们将数据 normalize 到了 [-1, 1]，现在要变回 [0, 1] 以便显示
    # 公式：img = img * std + mean -> img = img * 0.5 + 0.5 -> 等价于 img/2 + 0.5
    img = img / 2 + 0.5

    # 将 PyTorch Tensor 转换为 NumPy 数组，因为 matplotlib 处理的是 NumPy 数组
    npimg = img.numpy()

    # 调整数组维度顺序以便显示
    # PyTorch 格式是 (Channel, Height, Width) -> (3, 32, 32)
    # Matplotlib 需要 (Height, Width, Channel) -> (32, 32, 3)
    # np.transpose 将维度从 (0,1,2) 变换为 (1,2,0)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # 显示图像
    # 因为前面开了 plt.ion()，这里不会卡住程序
    plt.show()


# 定义主函数，封装整个训练和测试流程
def main():
    # --- 1. 数据准备部分 ---

    # 定义数据预处理/增强管道 (Compose 将多个变换串联起来)
    transform = transforms.Compose([
        # 第一步：将 PIL 图片或 NumPy 数组转换为 PyTorch Tensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
        transforms.ToTensor(),
        # 第二步：标准化 (Normalize)
        # 参数1: mean (均值)，参数2: std (标准差)
        # 对 R, G, B 三个通道分别减去 0.5 再除以 0.5
        # 结果是将数据分布从 [0, 1] 映射到 [-1, 1]，均值为 0，方差为 1，利于神经网络收敛
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载 CIFAR-10 训练集
    # download=True: 如果本地没有数据则自动下载
    # root="./data": 数据存放目录
    # train=True: 加载训练集 (False 则为测试集)
    # transform=transform: 应用上面定义的预处理
    # 【修正】类名应为 torchvision.datasets.CIFAR10 (注意大小写，原代码 CIFAR10 会报错)
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    # 创建训练数据加载器 (DataLoader)
    # trainset: 数据集对象
    # batch_size=4: 每个批次读取 4 张图片
    # shuffle=True: 每个 epoch 开始时打乱数据顺序，防止模型记忆顺序
    # num_workers=2: 使用 2 个子进程并行加载数据，加速 IO (macOS 上有时设为 0 更稳定)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # 加载 CIFAR-10 测试集 (逻辑同上，只是 train=False)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # 创建测试数据加载器
    # 测试时通常不需要 shuffle，但这里保留了 shuffle=True 也无妨
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # 定义 CIFAR-10 的 10 个类别标签名称
    # 注意：原代码中有拼写错误 ('brid'->'bird', 'forg'->'frog')，这里已修正，否则打印结果会奇怪
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- 2. 定义神经网络模型 ---

    # 定义一个名为 Net 的类，继承自 nn.Module (所有 PyTorch 模型的基类)
    class Net(nn.Module):
        # 初始化函数：定义网络层
        def __init__(self):
            # 调用父类 (nn.Module) 的构造函数，必须写
            super(Net, self).__init__()

            # 定义第一个卷积层
            # in_channels=3 (RGB 图片), out_channels=6 (输出 6 个特征图), kernel_size=5 (卷积核 5x5)
            self.conv1 = nn.Conv2d(3, 6, 5)

            # 定义最大池化层
            # kernel_size=2, stride=2 (默认)，即 2x2 窗口，步长为 2，将尺寸减半
            # 【修正】类名应为 nn.MaxPool2d (原代码 MaxPool2d 拼写错误)
            self.pool = nn.MaxPool2d(2, 2)

            # 定义第二个卷积层
            # 输入 6 通道 (上一层的输出), 输出 16 通道, 卷积核 5x5
            self.conv2 = nn.Conv2d(6, 16, 5)

            # 定义全连接层 (线性层)
            # 输入特征数计算：
            # 原始图片 32x32 -> conv1(5x5) -> 28x28 -> pool -> 14x14
            # 14x14 -> conv2(5x5) -> 10x10 -> pool -> 5x5
            # 此时有 16 个通道，所以展平后向量长度为 16 * 5 * 5 = 400
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入 400, 输出 120
            self.fc2 = nn.Linear(120, 84)  # 输入 120, 输出 84
            self.fc3 = nn.Linear(84, 10)  # 输入 84, 输出 10 (对应 10 个分类)

        # 前向传播函数：定义数据如何流经网络
        def forward(self, x):
            # 1. 卷积 -> ReLU 激活 -> 池化
            # F.relu 是函数式接口，self.conv1(x) 是层对象调用
            x = self.pool(F.relu(self.conv1(x)))

            # 2. 第二次 卷积 -> ReLU 激活 -> 池化
            x = self.pool(F.relu(self.conv2(x)))

            # 3. 展平 (Flatten)
            # 将多维特征图展平为一维向量，以便输入全连接层
            # -1 表示自动推断批次大小 (batch_size)
            x = x.view(-1, 16 * 5 * 5)

            # 4. 全连接层 -> ReLU 激活
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            # 5. 输出层 (通常分类任务最后一层不加激活函数，CrossEntropyLoss 内部会处理 Softmax)
            x = self.fc3(x)

            # 返回最终输出 (logits)
            return x

    # 实例化网络模型
    net = Net()

    # (可选) 如果要用 GPU/MPS 加速，需在此处将模型移动到设备: net.to(device)
    # 原代码注释掉了设备相关代码，这里保持 CPU 运行

    # --- 3. 定义损失函数和优化器 ---

    # 定义损失函数：交叉熵损失 (适用于多分类问题)
    criterion = nn.CrossEntropyLoss()

    # 定义优化器：随机梯度下降 (SGD)
    # net.parameters(): 告诉优化器去更新哪些参数 (即 net 中的所有权重)
    # lr=0.001: 学习率，控制每次更新的步长
    # momentum=0.9: 动量，帮助加速收敛并抑制震荡
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # --- 4. 开始训练循环 ---

    # 外层循环：遍历整个数据集的次数 (Epochs)
    # 这里只训练 2 轮，实际应用中可能需要 10-50 轮或更多
    for epoch in range(2):

        # 初始化当前 epoch 的累积损失
        running_loss = 0.0

        # 内层循环：遍历训练集中的每一个批次 (mini-batch)
        # enumerate(trainloader, 0): 获取批次索引 i 和数据 data
        for i, data in enumerate(trainloader, 0):
            # 解包数据：inputs 是图像 tensor, labels 是对应的标签 tensor
            inputs, labels = data

            # (可选) 如果使用 GPU/MPS，需将数据移到设备上:
            # inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            # PyTorch 默认会累积梯度，所以在每次反向传播前必须手动清零
            optimizer.zero_grad()

            # --- 前向传播 ---
            # 将输入图片传入网络，得到预测输出
            outputs = net(inputs)

            # 计算损失：比较预测输出 outputs 和真实标签 labels
            loss = criterion(outputs, labels)

            # --- 反向传播 ---
            # 自动计算所有参数的梯度 (backward propagation)
            loss.backward()

            # --- 参数更新 ---
            # 根据计算出的梯度，利用优化器更新网络权重
            optimizer.step()

            # 统计损失
            # loss.item() 将 tensor 转换为 python 数值
            running_loss += loss.item()

            # 每 2000 个 mini-batch 打印一次平均损失
            # i % 2000 == 1999 意味着当 i 为 1999, 3999... 时执行 (即每 2000 次)
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                # 重置累积损失，为下一个统计周期做准备
                running_loss = 0.0

    # 训练结束提示
    print('Finished Training')

    # --- 5. 测试/评估部分 ---

    # 从测试加载器中获取一个迭代器
    dataiter = iter(testloader)
    # 获取下一个批次的数据 (images, labels)
    # 【修正】Python 3 中推荐使用 next(dataiter) 而不是 dataiter.__next__()，虽然两者都行
    images, labels = next(dataiter)

    # 显示这一批次的 4 张图片
    # torchvision.utils.make_grid 将 4 张 (3,32,32) 的图拼成一张大网格图
    imshow(torchvision.utils.make_grid(images))

    # 打印这 4 张图片的真实标签名称
    # '%5s' 格式化字符串，保证对齐
    print(' '.join('%5s' % classes[labels[i]] for i in range(4)))

    # 将图片输入网络进行预测
    outputs = net(images)

    # 获取预测结果
    # torch.max(outputs, 1) 返回两个值：(最大值, 最大值的索引)
    # 我们只需要索引 (_)，因为它代表了预测的类别 ID
    _, predicted = torch.max(outputs, 1)

    # 打印预测的类别名称
    print('Predicted:', ' '.join('%5s' % classes[predicted[i]] for i in range(4)))

    # --- 6. 在整个测试集上评估准确率 ---

    correct = 0  # 记录预测正确的总数
    total = 0  # 记录测试图片的总数

    # torch.no_grad() 上下文管理器
    # 告诉 PyTorch 不需要计算梯度 (测试阶段不需要反向传播)，可以节省内存并加速计算
    with torch.no_grad():
        # 遍历整个测试集
        for data in testloader:
            images, labels = data

            # 前向传播
            outputs = net(images)

            # 获取预测类别
            # outputs.data 获取 tensor 的数据部分 (在新版本中直接用 outputs 即可)
            _, predicted = torch.max(outputs.data, 1)

            # 累加总图片数 (labels.size(0) 等于 batch_size)
            total += labels.size(0)

            # 累加正确数
            # (predicted == labels) 生成一个布尔 tensor，sum() 求和得到正确个数
            # .item() 转为 python 数字
            correct += (predicted == labels).sum().item()

    # 打印整体准确率
    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

    # --- 7. 计算每个类别的准确率 ---

    # 初始化列表，存储每个类别的正确数和总数
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # 比较预测和真实标签，得到布尔数组 c (形状同 batch_size)
            # squeeze() 去除多余的维度 (如果有的话)
            c = (predicted == labels).squeeze()

            # 遍历当前批次中的每一张图片
            for i in range(4):  # 假设 batch_size 是 4
                label = labels[i]  # 获取真实标签 ID

                # 如果预测正确 (c[i] 为 True/1)，则对应类别的正确数 +1
                class_correct[label] += c[i].item()

                # 对应类别的总数 +1
                class_total[label] += 1

    # 打印每个类别的准确率
    for i in range(10):
        # 避免除以零错误 (虽然 CIFAR-10 每个类都有数据)
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print("Accuracy of %5s : %2d %%" % (classes[i], acc))
        else:
            print("Accuracy of %5s : No samples" % classes[i])


# 入口判断
if __name__ == '__main__':
    # 只有在直接运行此脚本时才执行 main()
    # 这非常重要：当 DataLoader 使用 num_workers > 0 时，会启动子进程重新导入此文件
    # 如果没有这个判断，子进程也会执行 main()，导致死循环或重复创建 DataLoader 崩溃
    main()
