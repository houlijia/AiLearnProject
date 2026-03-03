import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 参数设置
input_size = 10  # 输入特征数量
num_classes = 2  # 类别数
num_epochs = 20  # 训练轮次
batch_size = 64  # 批大小
learning_rate = 0.001  # 学习率

# 创建随机数据集
# 这里我们生成一些随机数据作为示例
x_train = torch.randn(1000, input_size)  # 1000个样本，每个样本有10个特征
y_train = torch.randint(0, num_classes, (1000,))  # 为每个样本分配一个类别标签

# 将数据封装到DataLoader中，便于批次训练
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # 全连接层

    def forward(self, x):
        out = self.fc(x)
        return out


model = SimpleNN(input_size=input_size, num_classes=num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        # 反向传播并优化
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("训练完成")