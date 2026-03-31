'''
Description  : 90天学习计划 - 第四天-函数
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''

def sayHi():
    print("Hi, I'm LinJie")

def add(a, b):
    return a + b
result = add(1, 2)
print(add(1, 2), f'result=>{result}')

sayHi()


def is_even(num):
    if num % 2 == 0:
        return True
    else:
        return False

print(f'is_even(1)=>{is_even(1)}')
print(f'is_even(2)=>{is_even(2)}')