'''
Description  : 90天学习计划 - 第5天-test
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-01
LastEditors  : linjie
LastEditTime : 2026-04-01
'''
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei","Arial Unicode MS"]

# 1. 创建数据
df = pd.DataFrame({
    "name": ["小明","小红","小刚"],
    "age": [20,21,22],
    "score": [85,75,90]
})

# 2. 筛选分数 >80
result = df[df["score"]>80]
print("高分同学：")
print(result)

# 3. 统计 + 画图
print("平均分：", df["score"].mean())
plt.bar(df["name"], df["score"])
plt.title("成绩分布图")
plt.show()

print("\n🎉 第一阶段全部通关！")
