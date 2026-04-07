'''
Description  : 90天学习计划 - 第8天-线性层、激活函数、损失函数
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-07
LastEditors  : linjie
LastEditTime : 2026-04-07
'''
import torch
import torch.nn as nn

# ========================
# 1. 线性层 Linear
# 相当于：y = weight * x + bias
# 前端理解：一个公式计算逻辑
# ========================

# 输入特征4个 → 输出特征8个
linear = nn.Linear(in_features=4, out_features=8)

# 造一个模拟输入：batch=2条数据，每条4个特征
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
], dtype=torch.float32)

# 过线性层
out = linear(x)
print("线性层输出形状:", out.shape)  # (2, 8)


# ========================
# 2. 激活函数 ReLU
# 作用：把负数变成0，增加模型表达能力
# 理解成一个“过滤器”
# ========================
relu = nn.ReLU()
after_relu = relu(out)
print("激活后形状:", after_relu.shape)


# ========================
# 3. 损失函数 CrossEntropyLoss（分类专用）
# 作用：计算“预测值”和“真实标签”差多少
# 分数越小，模型越准
# ========================
loss_fn = nn.CrossEntropyLoss()

# 模拟模型输出（2条数据，3分类）
pred = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
# 真实标签
label = torch.tensor([0, 1], dtype=torch.long)

loss = loss_fn(pred, label)
print("损失值:", loss.item())


print("\n🎉 Day8-4 内容全部跑完！")
# 定义函数
def add(a, b):
  return a + b
# 类里的方法就是“绑定在对象上的函数”
class Calculator:
  def add(self, a, b):
    return a + b
c = Calculator()
print(c.add(2, 3))