'''
Description  : 
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-02
LastEditors  : linjie
LastEditTime : 2026-04-02
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 读取数据
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. 选特征 + 处理空值
df["Age"] = df["Age"].fillna(df["Age"].mean())
X = df[["Pclass", "Age", "Sex"]]
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)
y = df["Survived"]

# 3. 拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 训练
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 评分
print("🎯 模型准确率：", accuracy_score(y_test, y_pred))
print("🎉 你的第一个 AI 模型训练完成！")