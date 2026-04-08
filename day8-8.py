'''
Description  : 90天学习计划 - 第8天-CIFAR10彩色图分类
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
# 1. 数据预处理（彩色图 32x32）
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR10（10类彩色图）
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ========================
# 2. CNN 模型（适配 3 通道彩色图）
# ========================
class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: 3 通道 (RGB)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # 32*8*8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32→16
        x = self.pool(self.relu(self.conv2(x)))  # 16→8
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN_CIFAR10()

# ========================
# 3. 损失 & 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 4. 训练 1 轮（演示）
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
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | 损失: {avg_loss:.4f} | 准确率: {acc:.2f}%")

print("\n🎉 Day8-8 完成！CIFAR10 彩色图分类训练成功")


# 列表、元组
# 列表：可修改
my_list = [1, 2, 3]
my_list[0] = 10

# 元组：不可修改
my_tuple = (1, 2, 3)
# my_tuple[0] = 10 会报错