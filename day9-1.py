'''
Description  : 90天学习计划 - 第9天-LSTM模型
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

# ========================
# 1. 构造简单序列数据
# ========================
# 批次大小8，序列长度10，特征维度16（模拟词向量）
batch_size = 8
seq_len = 10
input_dim = 16
hidden_dim = 32
num_classes = 2

# 随机模拟文本序列
x = torch.randn(batch_size, seq_len, input_dim)
y = torch.randint(0, num_classes, (batch_size,))

# ========================
# 2. 定义 LSTM 模型
# ========================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True  # 形状: (batch, seq, feature)
        )
        # 分类层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # out: (batch, seq_len, hidden_dim)
        out, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = out[:, -1, :]
        # 分类
        logits = self.fc(last_out)
        return logits

model = LSTMModel()

# ========================
# 3. 损失 & 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 4. 一次训练步
# ========================
optimizer.zero_grad()
outputs = model(x)
loss = criterion(outputs, y)
loss.backward()
optimizer.step()

print("LSTM 输出形状:", outputs.shape)
print("损失值:", loss.item())
print("\n🎉 Day9-1 完成！LSTM 模型正常运行")