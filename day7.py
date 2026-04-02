'''
Description  : 90天学习计划 - 第7天-机器学习-学习曲线和验证曲线
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-02
LastEditors  : linjie
LastEditTime : 2026-04-02
'''
# ===================== 【1】导入需要用到的库 =====================
# pandas：数据处理
import pandas as pd
# numpy：数值计算
import numpy as np
# matplotlib：画图
import matplotlib.pyplot as plt

# 学习曲线、验证曲线专用工具
from sklearn.model_selection import learning_curve, validation_curve
# 随机森林模型（我们之前一直在用）
from sklearn.ensemble import RandomForestClassifier
# 导入鸢尾花数据集
from sklearn.datasets import load_iris

# ===================== 【2】Mac 画图不报错、不乱码 =====================
plt.rcParams["font.family"] = ["Arial Unicode MS", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 【3】加载数据 =====================
# 直接加载鸢尾花数据集（不需要自己找文件）
iris = load_iris()

# X：特征（4个：花萼长、宽；花瓣长、宽）
X = iris.data

# y：标签（0/1/2 代表三种花）
y = iris.target

# ===================== =====================
# 📉 第一部分：画 学习曲线 Learning Curve
# 作用：看模型是 欠拟合 / 刚好 / 过拟合
# ===================== =====================

print("正在生成学习曲线，请稍等...")

# learning_curve 会自动：
# 1. 用不同数量的训练集训练模型
# 2. 计算训练集分数
# 3. 计算验证集分数
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(),  # 模型：随机森林
    X, y,                      # 数据
    cv=5,                      # 5折交叉验证（更稳定）
    train_sizes=np.linspace(0.1, 1.0, 10)  # 训练集从10%→100%，取10个点
)

# 对多次交叉验证的结果求平均值（让曲线更平滑）
train_mean = train_scores.mean(axis=1)  # 训练集平均分数
val_mean = val_scores.mean(axis=1)      # 验证集平均分数

# 开始画图
plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_mean, label="训练集分数", color="blue")
plt.plot(train_sizes, val_mean, label="验证集分数", color="red")
plt.title("学习曲线（判断过拟合/欠拟合）")
plt.xlabel("使用的训练样本数量")
plt.ylabel("准确率")
plt.legend()  # 显示图例
plt.grid()    # 显示网格
plt.show()

# ===================== =====================
# 📈 第二部分：画 验证曲线 Validation Curve
# 作用：找到最好的超参数 max_depth（树最大深度）
# ===================== =====================

print("正在生成验证曲线，请稍等...")

# 我们要测试的 max_depth 范围
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# validation_curve 会自动：
# 1. 用不同的 max_depth 训练模型
# 2. 记录训练集分数
# 3. 记录验证集分数
train_scores, val_scores = validation_curve(
    RandomForestClassifier(),
    X, y,
    param_name="max_depth",    # 要调整的参数：树深度
    param_range=param_range,   # 参数范围
    cv=5
)

# 求平均
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# 画图
plt.figure(figsize=(10, 5))
plt.plot(param_range, train_mean, label="训练集", color="blue")
plt.plot(param_range, val_mean, label="验证集", color="red")
plt.title("验证曲线：寻找最佳 max_depth")
plt.xlabel("max_depth 值")
plt.ylabel("准确率")
plt.legend()
plt.grid()
plt.show()

print("✅ Day7 学习曲线 + 验证曲线 运行完成！")
# 遍历字典
student = {"name": "小明", "age": 20, "major": "AI"}
for k, v in student.items():
    print(k, ":", v)