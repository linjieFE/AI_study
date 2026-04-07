'''
Description  : 数据集 Dataset + 数据加载器 DataLoader
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-07
LastEditors  : linjie
LastEditTime : 2026-04-07
'''
# ========================
# Day8-3 数据集 Dataset + 数据加载器 DataLoader
# ========================
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ========================
# 1. 自定义数据集类（固定模板）
# ========================
class MyDataset(Dataset):
    # 初始化：传入特征 X 和标签 y
    def __init__(self, X, y):
        # 转成 float32 类型，适配训练
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    # 返回总样本数
    def __len__(self):
        return len(self.X)

    # 根据索引取一条数据
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========================
# 2. 构造模拟数据（也可以换成真实数据）
# ========================
# 100个样本，每个样本4个特征（类似鸢尾花）
X = np.random.randn(100, 4)
# 100个标签，0/1/2 三类
y = np.random.randint(0, 3, size=100)

# ========================
# 3. 创建数据集实例
# ========================
dataset = MyDataset(X, y)

# 查看数据集长度
print("数据集总样本数：", len(dataset))

# 取一条数据看看
sample_X, sample_y = dataset[0]
print("\n第一条样本特征：", sample_X)
print("第一条样本标签：", sample_y)

# ========================
# 4. 创建 DataLoader（批量加载）
# ========================
dataloader = DataLoader(
    dataset,
    batch_size=8,    # 每批8个数据
    shuffle=True,    # 打乱顺序，防止模型学顺序
    num_workers=0    # Mac 一般设0不报错
)

# ========================
# 5. 遍历一个批次（训练时就是这么循环）
# ========================
print("\n=== 遍历一个批次数据 ===")
for batch_X, batch_y in dataloader:
    print("批次 X 形状：", batch_X.shape)  # (8,4)
    print("批次 y 形状：", batch_y.shape)  # (8,)
    break  # 只看第一个批次

print("\n🎉 Day8-3 完成！数据集 & 加载器使用正常")