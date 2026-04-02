'''
Description  : 90天学习计划 - 第6天-机器学习-特征重要性
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-02
LastEditors  : linjie
LastEditTime : 2026-04-02
'''
# 导入类型注解库
from typing import Any

# 导入pandas库
import pandas as pd
# 导入训练集和测试集拆分函数
from sklearn.model_selection import train_test_split
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 导入准确率评分函数
from sklearn.metrics import accuracy_score
# 导入绘图库
import matplotlib.pyplot as plt
# 解决中文显示问题，防止乱码
# 替换成 Mac 专用字体设置
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False


# 1. 数据处理
# 读取数据
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# 读取数据

# 填充年龄空值
# df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Age"] = df["Age"].fillna(df["Age"].median())

# 选择特征
X = df[["Pclass", "Age", "Sex"]]
# 独热编码，将Sex转换为0和1
# drop_first=True，删除第一个类别，避免多重共线性
# columns=["Sex"]，指定要独热的列
# 独热编码后，Sex变成了Sex_male和Sex_female两个新列
# 这里我们只保留Sex_male，删除Sex_female
# 这样我们就可以用Sex_male来表示性别了
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

# 选择目标变量
y = df["Survived"]

# 2. 训练模型
# 拆分训练集和测试集
# 随机种子，保证每次运行结果相同
# test_size=0.2，测试集占20%
# random_state=42，随机种子，保证每次运行结果相同
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''

随机森林分类器
随机森林分类器是一种集成学习方法，通过构建多个决策树来提高模型的准确性
'''

model = RandomForestClassifier()
# 训练模型
model.fit(X_train, y_train)

# 3. 评分
y_pred = model.predict(X_test)
print("模型准确率：", accuracy_score(y_test, y_pred))

# ===================== 核心：看特征重要性 =====================
importances = model.feature_importances_
features = X.columns

# 打印每个特征的重要程度
print("\n【特征重要性】")
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.2%}")

# 画图展示
plt.bar(features, importances)
plt.title("各因素对生还的影响程度")
plt.ylabel("重要性")
plt.show()