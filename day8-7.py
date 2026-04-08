'''
Description  : 90天学习计划 - 第8天-MNIST手写数字识别-CNN模型
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-07
LastEditors  : linjie
LastEditTime : 2026-04-07
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========================
# 1. 数据预处理（MNIST手写数字）
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ========================
# 2. 定义CNN模型（核心）
# ========================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1：输入1通道（灰度图），输出16个特征图，卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 激活函数
        self.relu = nn.ReLU()
        # 池化层：2x2窗口，步长2，把图片尺寸缩小一半
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2：输入16通道，输出32个特征图
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 全连接层：把32*7*7的特征图展平，输出128维
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # 输出层：10分类（0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入x形状：[batch, 1, 28, 28]
        x = self.conv1(x)  # → [64, 16, 28, 28]
        x = self.relu(x)
        x = self.pool(x)   # → [64, 16, 14, 14]（尺寸减半）
        
        x = self.conv2(x)  # → [64, 32, 14, 14]
        x = self.relu(x)
        x = self.pool(x)   # → [64, 32, 7, 7]（再减半）
        
        # 展平：[64, 32, 7, 7] → [64, 32*7*7=1568]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)    # → [64, 128]
        x = self.relu(x)
        x = self.fc2(x)    # → [64, 10]
        return x

# 初始化模型
model = CNNModel()

# ========================
# 3. 损失函数 + 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 4. 训练循环（1轮演示）
# ========================
epochs = 1
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 累计损失和准确率
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | 平均损失: {avg_loss:.4f} | 训练准确率: {acc:.2f}%")

print("\n🎉 Day8-7 完成！CNN 模型训练成功 ✅")

# 补：Day4 字典（深度学习配置常用）
# 字典：键值对，类似前端的对象
config = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 5
}
print("学习率:", config["lr"])

# 集合：去重
nums = [1, 2, 2, 3, 3, 3]
unique_nums = set(nums)
print("去重后:", unique_nums)