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
# 1. 创建测试数据（含空值）
data = {
 "name": ["张三", "李四", "王五", "赵六"],
 "age": [25, None, 28, 32], # None = 空值
 "salary": [12000, 15000, 9000, 25000]
}
df = pd.DataFrame(data)
print("【原始数据】")
print(df.to_string(index=False))

# 2. 删除空值
df_clean = df.dropna()
# 查看有没有空值
print("\n【是否为空值】")
print(df_clean.isna().sum().to_string())

# 方案A：删除有空值的行
df_drop = df.dropna()
print("\n【删除空值】")
print(df_drop.to_string(index=False))

# 方案B：填充空值
df_fill = df.fillna({"age": 30}) # 年龄空的填30
print("\n【填充空值】")
print(df_fill.to_string(index=False))

# 筛选：工资 > 10000 或 年龄 < 30
# 打印美化表格
result = df[(df["salary"] > 10000) | (df["age"] < 30)]
print("\n【筛选：工资 > 10000 或 年龄 < 30】")
# print(result.to_string(index=False, justify='left'));
print
print(result.to_markdown(index=False,tablefmt="grid"))