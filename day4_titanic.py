'''
Description  : 
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''
import pandas as pd

# 直接读取在线数据集（不用你下载！）
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ===================== 1. 看数据整体情况 =====================
print("【数据前5行】")
print(df.head().to_markdown(index=False, tablefmt='grid'))

print("\n【数据信息】")
df.info()

print("\n【基础统计】")
print(df.describe().to_markdown(index=False, tablefmt='grid'))

# ===================== 2. 简单分析 =====================
# 多少人生还（1=生还，0=死亡）
survived_count = df["Survived"].value_counts()
print("\n【生还/死亡人数】")
print(survived_count.to_markdown(index=False, tablefmt='grid'))

# 生存率
survived_rate = df["Survived"].mean()
print(f"\n【整体生存率】: {survived_rate:.2%}")