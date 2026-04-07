'''
Description  : 
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-07
LastEditors  : linjie
LastEditTime : 2026-04-07
'''
# ========================
# Day8-1 PyTorch 安装测试
# 作用：验证 PyTorch 安装成功
# ========================

# 导入 PyTorch
import torch

# 查看版本
print("PyTorch 版本:", torch.__version__)

# 查看是否可以使用 CPU（肯定可以）
print("是否可用 CPU:", torch.backends.mps.is_available() or True)

# 创建一个简单张量（类似数组/矩阵）
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 做个简单加法
z = x + y
print("x + y =", z)

print("\n🎉 Day8-1 完成！PyTorch 安装并运行成功！")