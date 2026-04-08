'''
Description  : 90天学习计划 - 第9天-LSTM文本分类
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-08
LastEditors  : linjie
LastEditTime : 2026-04-08
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========================
# 1. 构造模拟文本数据（演示用）
# 真实项目只要替换成自己的数据集即可
# ========================
# 假设我们有 100 条文本，每条文本固定 20 个词向量
num_samples = 100
seq_len = 20
embed_dim = 16
num_classes = 2  # 二分类：正面/负面

# 模拟文本特征（可以理解成词向量序列）
text_data = torch.randn(num_samples, seq_len, embed_dim)
# 模拟标签：0=负面，1=正面
labels = torch.randint(0, num_classes, (num_samples,))

# ========================
# 2. 封装成 Dataset（固定模板）
# ========================
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

dataset = TextDataset(text_data, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ========================
# 3. 定义 LSTM 文本分类模型
# ========================
class LSTM_TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 32  # LSTM 隐藏层维度

        # LSTM 层：输入维度=embed_dim，输出=hidden_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 分类层：取最后一个时间步 → 输出 2 类
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出作为句子特征
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        logits = self.fc(last_out)    # (batch, num_classes)
        return logits

model = LSTM_TextClassifier()

# ========================
# 4. 损失函数 & 优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 5. 训练一步（演示）
# ========================
epochs = 1
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
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

print("\n🎉 Day9-2 完成！LSTM 文本分类模型训练成功")