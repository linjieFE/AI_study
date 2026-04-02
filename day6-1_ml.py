'''
Description  : 90天学习计划 - 第6天-机器学习-手动预测
Version      : v1.0
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

# 1. 数据读取与处理
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df["Age"] = df["Age"].fillna(df["Age"].mean())
X = df[["Pclass", "Age", "Sex"]]
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)
y = df["Survived"]

# 2. 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. 测试评分
y_pred = model.predict(X_test)
print("模型准确率：", accuracy_score(y_test, y_pred))

# ===================== 无警告预测 =====================
# 构造和训练时一样的列名格式
import pandas as pd

def predict_survival(pclass, age, sex_male):
    data = {
        "Pclass": [pclass],
        "Age": [age],
        "Sex_male": [sex_male]
    }
    df_person = pd.DataFrame(data)
    result = model.predict(df_person)
    return "生还" if result[0] == 1 else "死亡"

# 开始预测
print("\n30岁男性，1等舱：", predict_survival(1, 30, 1))
print("25岁女性，3等舱：", predict_survival(3, 25, 0))