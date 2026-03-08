import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 参数设置
input_size = 10
num_classes = 2
num_epochs = 20
batch_size = 64
learning_size = 0.001


# 创建随机数据集
x_train = torch.randn(1000, input_size)
y_train = torch.randint(0, num_classes, (1000,))

# 将数据封装在Dataloader中
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

model = SimpleNN(input_size=input_size, num_classes=num_classes)

## 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_size)


# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
