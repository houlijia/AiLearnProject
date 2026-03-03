## 代码解析

    导入库:
        torch: PyTorch 的核心库。
        torch.nn: 包含所有用于构建神经网络层的模块。
        torch.utils.data: 提供数据加载工具，DataLoader 和 TensorDataset 是其中的核心组件。
    参数设置:
        定义了模型和训练过程中的超参数，如输入维度、类别数、训练轮数等。这种集中管理参数的方式非常清晰。
    数据准备:
        x_train = torch.randn(1000, input_size): 创建了一个形状为 (1000, 10) 的张量，代表 1000 个样本，每个样本有 10 个随机特征值。
        y_train = torch.randint(0, num_classes, (1000,)): 创建了一个形状为 (1000,) 的张量，包含 1000 个随机标签（0 或 1），作为目标值。
        TensorDataset: 将特征张量 x_train 和标签张量 y_train 组合成一个数据集对象。
        DataLoader: 接收 TensorDataset，并将其打包成可迭代的批次（batch_size=64），shuffle=True 确保每次训练时数据顺序是打乱的，这有助于模型收敛。
    模型定义 (SimpleNN):
        继承自 nn.Module，这是所有 PyTorch 网络的基类。
        __init__: 在构造函数中定义了模型的所有层。这里只有一个全连接层 (nn.Linear)，它接收 input_size 个输入特征，并输出 num_classes 个值（对应两个类别的得分）。
        forward: 定义了数据如何通过这些层。这里是纯线性的，没有激活函数。
    损失函数与优化器:
        nn.CrossEntropyLoss(): 适用于多分类问题的标准损失函数。它内部结合了 LogSoftmax 和 NLLLoss，非常适合最后一层输出未经过归一化的 logits 的情况。
        torch.optim.Adam(): Adam 是一种非常流行且高效的优化器，能够自适应地调整学习率。
    训练循环:
        外层循环 for epoch in range(num_epochs): 遍历整个数据集 num_epochs 次。
        内层循环 for i, (inputs, labels) in enumerate(train_loader): 遍历 DataLoader 提供的每一个批次数据。
        outputs = model(inputs): 将输入数据送入模型进行预测。
        loss = criterion(...): 计算模型预测值和真实标签之间的差距。
        optimizer.zero_grad(): 在每次反向传播前清零梯度，防止梯度累积。
        loss.backward(): 执行反向传播，计算损失相对于模型参数的梯度。
        optimizer.step(): 根据计算出的梯度更新模型参数。
        print(...): 每隔 10 个批次打印一次当前的损失，用于监控训练进度。