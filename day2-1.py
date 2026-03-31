'''
Description  : 90天学习计划 - 第三天
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-30
LastEditors  : linjie
LastEditTime : 2026-03-31
'''
# 条件语句

score = 80
if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 70:
    print("中等")
else:
    print("不及格")

# 循环语句
names = ["张三", "李四", "王五"]
for i in names:
    print(f"姓名=>{i}")
# 循环数字
for i in range(5):
    print(f"for=>{i}")
# while
i = 0
while i < 5:
    print(f"while=>{i}")
    i += 1
#
for i in range(1, 10):
    if i % 2 == 0:
        print(f"这是偶数=>{i}")
        continue
    else:
        print(f"这是奇数=>{i}")