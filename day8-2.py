'''
Description  : 90天学习计划 - 第8天-Tensor 基础与自动求导
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-07
LastEditors  : linjie
LastEditTime : 2026-04-07
'''
# ========================
# Day8-2 Tensor 基础与自动求导
# ========================
import torch

# ========================
# 1. Tensor 创建（最基本用法）
# ========================
# 从列表创建 tensor
a = torch.tensor([1.0, 2.0, 3.0])
# 全0矩阵
b = torch.zeros((2, 3))
# 全1矩阵
c = torch.ones((2, 2))
# 随机矩阵
d = torch.randn(3, 3)

print("=== 基础 Tensor ===")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)

# ========================
# 2. 常用属性
# ========================
print("\n=== Tensor 属性 ===")
print("a 的形状 shape:", a.shape)
print("a 的数据类型 dtype:", a.dtype)

# ========================
# 3. 基本运算
# ========================
x = torch.tensor([2.0])
w = torch.tensor([3.0], requires_grad=True)  # 需要计算梯度
b = torch.tensor([1.0], requires_grad=True)

# 简单函数 y = w*x + b
y = w * x + b
print("\n=== 运算结果 ===")
print("y =", y)

# ========================
# 4. 自动求导（核心）
# ========================
# 反向传播，计算梯度
y.backward()

# 查看梯度
print("\n=== 自动求导结果 ===")
print("w 的梯度 dw =", w.grad)  # 应该等于 x = 2
print("b 的梯度 db =", b.grad)  # 应该等于 1

print("\n🎉 Day8-2 完成！")

import os

# 查看当前路径
print("当前路径:", os.getcwd())

# 创建新文件夹
if not os.path.exists("model_save"):
    os.mkdir("model_save")
    print("已创建 model_save 文件夹")