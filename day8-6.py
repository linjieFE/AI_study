'''
Description  : 90天学习计划 - 第8天-MNIST手写数字识别
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
# 1. 图片预处理：转成张量 + 归一化
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========================
# 2. 下载/加载 MNIST 数据集
# ========================
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ========================
# 3. 构建模型（识别手写数字）
# ========================
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 把 28x28 图片展平成 784
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 0~9 共10类

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MNISTModel()

# ========================
# 4. 损失 & 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 5. 训练（只训练 1 轮方便演示）
# ========================
epochs = 1
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | 损失: {avg_loss:.4f} | 准确率: {acc:.2f}%")

print("\n🎉 Day8-7 完成！MNIST 手写数字识别训练成功")