卷积神经网络（CNN）是深度学习中用于处理图像、视频等具有网格结构数据的核心模型。在构建和训练 CNN 时，会用到一系列关键**函数/组件**，它们各自承担不同作用，并在特定阶段使用。以下是系统化的分类说明：

---

## 一、核心结构函数（用于搭建网络）

| 函数/层 | 作用 | 关键参数 | 使用时机 |
|--------|------|--------|--------|
| **卷积层（Convolutional Layer）**如 `torch.nn.Conv2d` / `tf.keras.layers.Conv2D` | 提取局部特征（边缘、纹理、形状等） | 卷积核大小（3×3, 5×5）、步长（stride）、填充（padding）、输出通道数 | **每一层特征提取开始时**，通常堆叠多个卷积层 |
| **激活函数（Activation Function）**如 `ReLU`, `LeakyReLU`, `Sigmoid` | 引入非线性，使网络能拟合复杂函数 | — | **紧跟在卷积层或全连接层之后**• 中间层：优先用 **ReLU**• 深层/防“死亡神经元”：用 **Leaky ReLU**• 二分类输出：用 **Sigmoid**• 多分类输出：用 **Softmax**（通常在最后） |
| **池化层（Pooling Layer）**如 `MaxPool2d`, `AvgPool2d` | 下采样，降低空间维度，减少计算量，增强平移不变性 | 池化窗口大小（如 2×2）、步长 | **通常在卷积+激活后使用**，用于压缩特征图尺寸 |
| **全连接层（Fully Connected / Dense）**如 `torch.nn.Linear` / `keras.layers.Dense` | 整合全局特征，进行最终分类或回归 | 输出维度（如类别数） | **网络末端**，将展平后的特征映射到输出空间 |

---

## 二、辅助优化函数（提升训练效果）

| 函数/层 | 作用 | 使用时机 |
|--------|------|--------|
| **批归一化（Batch Normalization, BN）**如 `nn.BatchNorm2d` | 对每一批特征图归一化（均值0，方差1），缓解内部协变量偏移，加速收敛，提升稳定性 | **通常放在卷积层之后、激活函数之前**（如 Conv → BN → ReLU） |
| **Dropout**如 `nn.Dropout(p=0.5)` | 随机“关闭”部分神经元，防止过拟合 | **常用于全连接层之间**，也可用于深层卷积层；**仅在训练时启用** |
| **展平层（Flatten）** | 将三维特征图 `[H, W, C]` 转为一维向量 | **在最后一个卷积/池化层之后、第一个全连接层之前** |

---

## 三、训练与评估函数

| 函数类型 | 常见实现 | 作用 | 使用时机 |
|--------|--------|------|--------|
| **损失函数（Loss）** | • 分类：`CrossEntropyLoss`• 回归：`MSELoss` | 衡量预测值与真实标签的差距 | **训练过程中每个 batch 计算一次**，用于反向传播 |
| **优化器（Optimizer）** | `SGD`, `Adam`, `RMSprop` | 根据损失梯度更新模型参数 | **训练循环中，每次反向传播后调用 `.step()`** |
| **评估指标** | 准确率（Accuracy）、混淆矩阵、F1-score | 评估模型在验证/测试集上的性能 | **训练后或每个 epoch 结束时计算** |

---

## 四、典型使用流程（以图像分类为例）

```python
# 1. 输入图像 (e.g., 224×224×3)
x = input_image

# 2. 特征提取阶段（重复多次）
x = Conv2d(x)        # 提取特征
x = BatchNorm2d(x)   # 稳定分布（可选但推荐）
x = ReLU(x)          # 引入非线性
x = MaxPool2d(x)     # 降维

# 3. 展平 + 分类头
x = Flatten(x)       # [H, W, C] → [H*W*C]
x = Linear(x)        # 全连接
output = Softmax(x)  # 多分类概率（或直接用 CrossEntropyLoss，内部含 Softmax）

# 4. 训练时
loss = CrossEntropyLoss(output, label)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 5. 评估时
acc = accuracy(output, label)
```

---

## 五、选择建议总结

- **默认激活函数**：用 **ReLU**（快、稳、抗梯度消失）
- **防过拟合**：加 **Dropout**（全连接层） + **数据增强**
- **加速训练**：加 **BatchNorm**
- **深层网络**（>20层）：考虑 **残差连接（ResNet）**，避免退化
- **移动端部署**：用 **Depthwise Separable Conv**（如 MobileNet）减少参数

---
