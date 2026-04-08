'''
Description  : 90天学习计划 - 第8天-CIFAR10彩色图分类-Dropout防过拟合
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-07
LastEditors  : linjie
LastEditTime : 2026-04-08
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========================
# 1. 数据加载（CIFAR10）
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ========================
# 2. 带 Dropout 的 CNN 模型
# ========================
class CNN_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # 全连接层 + Dropout
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)  # 随机丢弃50%神经元
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 只在训练时生效
        x = self.fc2(x)
        return x

model = CNN_Dropout()

# ========================
# 3. 损失 & 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 4. 训练一轮
# ========================
epochs = 1
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    model.train()  # 训练模式：Dropout 开启

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

print("\n🎉 Day8-9 完成！Dropout 防过拟合模型训练成功")