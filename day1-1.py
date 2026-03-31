'''
Description  : 90天学习计划 - 第一天-字典
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-30
LastEditors  : linjie
LastEditTime : 2026-03-30
'''

# 列表
names = ["张三", "李四", "王五"]
# 访问（下标从 0 开始）
print(names[0])
print(names[1])
# 修改
names[1] = "赵六"
print(names)
# 添加
names.append("钱七")
print(names)
# 删除
del names[0]
print(names)