'''
Description  : 90天学习计划 - 第三天-pandas
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''
import pandas as pd

data = {
    "name": ["张三", "李四", "王五"],
     "age": [22, 25, 30],
     "score": [80, 90, 85]
}
df = pd.DataFrame(data)

print(f'df.head()=>{df.head()}')  # 前几行
print(f'df.info()=>{df.info()}')# 结构信息
print(f'df.describe()=>{df.describe()}') # 统计：均值、最大最小等
print(f'df.columns=>{df.columns}') # 列名

print(f'df["age"]=>{df["age"]}')
print(f'df[["name", "score"]]=>{df[["name", "score"]]}')
# 👇 就加这一行！全局美化 Pandas 打印表格
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

example = {
    "name": ["张三", "李四", "王五"], # 姓名
    "salary": [10000, 20000, 30000], # 工资
    "DataFrame": [df, df, df] # 数据框
}
df=pd.DataFrame(example)

format = df[df["salary"]>10000]
print(format.to_markdown(index=False,tablefmt="grid"))