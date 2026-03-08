import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # 开启交互模式


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # print(f"Using device: {device}")

    # transform 表示要转换
    # 神经网络特别钟爱经过标准化处理后的数据。标准化处理指的是data减去它的均值，
    # 再除它的标准差，最终data将呈现均值为零0方差为1的数据分布
    # 我们将其转化为tensor数量，并归一化为[-1, 1].
    transform = transforms.Compose([transforms.ToTensor(),  # PIL IMAGE -> TENSOR
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 0,1 -> -1,1
                                    ])
    trainset = torchvision.datasets.CIFAR10(download=True, root="./data", train=True, transform=transform)
    # 包装数据
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(download=True, root="./data", train=False, transform=transform)
    # 包装数据
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'brid', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck')

    # 查看图片
    # for i, data in enumerate(trainloader, 0):
    #     images, labels = data
    #     if i == 0:
    #         imshow(torchvision.utils.make_grid(images))
    #         break
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # print(''.join('%5s' % classes[labels[i]] for i in range(4)))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    # net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # 将数据移动到MPS设备
            # inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 验证/评估
    dataiter = iter(testloader)
    images, labels = dataiter.__next__()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[i]] for i in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)  # torch.topk(1)
    print('predicted:' + ' '.join('%5s' % classes[predicted[i]] for i in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # torch.topk(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy %d %%" % (100 * correct / total))

    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze() # 1*4 -> 4
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print("Accuracy of %5s: %2d %%" % (classes[i], 100*class_correct[i]/class_total[i]))


if __name__ == '__main__':
    # 只有当直接运行此脚本时，才执行 main 函数
    # 这样子进程导入该文件时，不会执行 main()，也就不会重复创建 DataLoader
    main()
