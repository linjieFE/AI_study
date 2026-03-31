'''
Description  : 
Version      : v1.0
Company      : shanghai
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''
import pandas as pd

data = {
    "name": ["张三", "李四", "王五", "李四", "小明"],
    "age": [25, 26, 28, 26, 24],
    "salary": [12000, 15000, 9000, 15000, 18000]
}
df = pd.DataFrame(data)

# 排序（最常用）
df_sorted = df.sort_values(by="salary", ascending=False)
print(" [按工资降序]\n")
print(df_sorted.to_markdown(index=False,tablefmt="grid"))

# 按名字去重
df_unique = df.drop_duplicates(subset=["name"])
print("\n【按名字去重】")
print(df_unique.to_markdown(index=False,tablefmt="grid"))

print("\n【统计信息】")
print("平均工资:", df['salary'].mean())
print("最大工资:", df['salary'].max())
print("最小工资:", df['salary'].min())
print("工资总和:", df['salary'].sum())
print("人数：",df["name"].count())
print("工资标准差:", df['salary'].std())
print("工资方差:", df['salary'].var())
print("工资中位数:", df['salary'].median())
print("工资众数:", df['salary'].mode())
print("工资分位数:", df['salary'].quantile(0.75))


# 新增等级列
df["level"] = df["salary"].apply(lambda x: "高收入" if x > 15000 else "普通")
print("\n【新增等级列】")
print(df.to_markdown(index=False,tablefmt="grid"))

# 按年龄升序
df_age_sorted = df.sort_values(by="age", ascending=True)
print("\n【按年龄升序】")
print(df_age_sorted.to_markdown(index=False,tablefmt="grid"))
