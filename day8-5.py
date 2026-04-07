'''
Description  : 90天学习计划 - 第8天-训练循环
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
from torch.utils.data import Dataset, DataLoader

# ========================
# 1. 构造假数据（模拟数据集）
# ========================
class SimpleDataset(Dataset):
    def __init__(self):
        # 100条数据，每条4个特征
        self.X = torch.randn(100, 4)
        # 标签：0/1/2 三分类
        self.y = torch.randint(0, 3, (100,))

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载器
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ========================
# 2. 定义简单模型
# ========================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)   # 输入4 → 隐藏8
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)   # 输出3分类

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel()

# ========================
# 3. 损失函数 + 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 4. 【核心】训练循环模板
# ========================
epochs = 5  # 训练5轮

for epoch in range(epochs):
    total_loss = 0.0

    # 遍历每一批数据
    for batch_X, batch_y in dataloader:
        # ① 梯度清零
        optimizer.zero_grad()

        # ② 前向传播
        outputs = model(batch_X)

        # ③ 计算损失
        loss = criterion(outputs, batch_y)

        # ④ 反向传播
        loss.backward()

        # ⑤ 更新参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

    # 打印一轮的平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.4f}")

print("\n🎉 训练完成！Day8-5 训练循环掌握 ✅")