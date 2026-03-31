'''
Description  : 
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''
from doctest import Example
import numpy as np

# 创建数组

arr =  np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

# 数组运算
a =  np.array([1,2,3])
b =  np.array([4,5,6])
print(a + b) #[5 7 9]
print(b - a) #[3 3 3]
print(a * b) #[4 10 18]
print(a / b) #[0.25 0.4 0.5]
print(arr ** 2) #[1 4 9 16 25]

arr1 = np.array([[1,2],[3,4],[5,6]])
print(f'取值（索引）arr1.shape=>{arr1.shape}') #三行两列(3, 2)
# 重塑形状
print(f'重塑arr1.reshape(2,3)=>{arr1.reshape(2,3)}') #变成两行三列[[1 2 3] [4 5 6]]
# 转置
print(f'转置arr1.T=>{arr1.T}') #转置后变成两行三列[[1 3 5] [2 4 6]]
# 展平
print(f'展平arr1.flatten()=>{arr1.flatten()}') # 展平后变成一维数组[1 2 3 4 5 6]
# 拼接
print(f'拼接np.concatenate((arr1, arr1))=>{np.concatenate((arr1, arr1))}') #拼接后变成两行四列[[1 2 3 4] [5 6 1 2] [3 4 5 6]]
# 分割
print(f'分割np.split(arr1, 2)=>{np.split(arr1, 1)}') #分割后变成两行一列[[1] [2] [3]]
# 堆叠
print(f'堆叠np.stack((arr1, arr1))=>{np.stack((arr1, arr1))}') #堆叠后变成两行三列[[1 2 3] [4 5 6] [1 2 3] [4 5 6]]
# 拆分
print(f'拆分np.split(arr1, 2)=>{np.split(arr1, 1, axis=1)}') #拆分后变成两行一列[[1] [2] [3]]

example = np.array([10, 20, 30, 40, 50])

print(example *3) #[30 60 90 120 150]