'''
Description  : 90天学习计划 - 第五天-matplotlib绘图
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-01
LastEditors  : linjie
LastEditTime : 2026-04-01
'''
import pandas as pd
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# 读取泰坦尼克数据
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ------------------- 1. 柱状图：男女数量 -------------------
sex_count = df["Sex"].value_counts()
plt.bar(sex_count.index, sex_count.values, color=["#ff9999","#66b3ff"])
plt.title("乘客性别数量")
plt.xlabel("性别")
plt.ylabel("人数")
plt.show()

# ------------------- 2. 饼图：生还 vs 死亡 -------------------
survived_count = df["Survived"].value_counts()
plt.pie(
    survived_count.values,
    labels=["死亡", "生还"],
    autopct="%.1f%%",
    colors=["#ff6666","#99ff99"]
)
plt.title("生还人数比例")
plt.show()

# ------------------- 3. 直方图：年龄分布 -------------------
plt.hist(df["age"].dropna(), bins=20, color="#c2c2f0")
plt.title("乘客年龄分布")
plt.xlabel("年龄")
plt.ylabel("人数")
plt.show()