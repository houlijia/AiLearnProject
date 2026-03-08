import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 设置镜像源
import os

os.environ['TORCH_HOME'] = './torch_home'


# 也可以直接指定数据集的下载 URL
# 例如，为 CIFAR-10 设置镜像
class CIFAR10WithMirror(torchvision.datasets.CIFAR10):
    base_url = "https://s3.amazonaws.com/fast-cifar-10/"
    resources = [
        ('https://gitee.com/yzy1996/cifar-10-python/raw/master/cifar-10-python.tar.gz',
         'c58f30108f718f92721af3b95e74348a'),
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)


# 参数设置
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 为了加速，我们直接使用 torchvision 的 CIFAR10，并且提前准备好数据
# 但先修改根目录，让数据下载到当前项目目录下
data_root = './data'

# 加载 CIFAR-10 数据集
# 如果你之前下载过，它会直接加载，不会重复下载
train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                             download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                            download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)


# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 输入通道数为3（RGB图像），输出通道数为16
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # 展平操作
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 检查数据集是否已存在，避免重复下载
try:
    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                                 download=True, transform=transform)
except Exception as e:
    print(f"自动下载失败: {e}")
    print("请手动下载数据集。")
    exit()

print("开始训练...")

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("训练完成")

# 测试模型
model.eval()  # 切换到评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'测试集上的准确率: {100 * correct / total:.2f}%')
