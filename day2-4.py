'''
Description  : 90天学习计划 - 第五天-异常处理
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''

try:
    print(10 / 0)
except ZeroDivisionError:
    print("不能除以0")
finally:
    print("异常处理完成")

try:
    with open('text1.txt',"r",encoding="utf-8") as f:
        content = f.read()
        print(content)
except Exception as e:
    print(f"读取文件失败: {e}")
finally:
    print("读取文件完成")

try:
    print("开始测试")
except:
    print("测试失败")
finally:
    print("测试完成")