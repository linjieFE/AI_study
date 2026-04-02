'''
Description  : 90天学习计划 - 第6天-机器学习-特征工程
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-02
LastEditors  : linjie
LastEditTime : 2026-04-02
'''
# 导入pandas库
import pandas as pd
# 导入numpy库
import numpy as np
# 导入matplotlib库
import matplotlib.pyplot as plt
# 导入训练集和测试集拆分函数
from sklearn.model_selection import train_test_split, cross_val_score
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 导入准确率评分函数
from sklearn.metrics import accuracy_score, classification_report
# 导入绘图库
import matplotlib.pyplot as plt
# 解决中文显示问题，防止乱码
# 替换成 Mac 专用字体设置
plt.rcParams["font.family"] = ["Arial Unicode MS", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 数据读取与高级清洗 =====================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1.1 填充缺失值（无警告版）
df["Age"] = df["Age"].fillna(df["Age"].median())  # 用中位数填充，更抗 outliers
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # 用众数填充登船港口
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# 1.2 构造新特征（特征工程核心）
# 家庭总人数 = 兄弟姐妹/配偶 + 父母/子女
# 1表示自己
# 2表示自己和配偶，家庭总人数为2
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
# 是否独自出行
# 如果家庭总人数为1，则独自出行
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
# 提取称谓（Mr/Mrs/Miss等，强特征）
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# 合并稀有称谓
df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].replace('Mlle', 'Miss')
df["Title"] = df["Title"].replace('Ms', 'Miss')
df["Title"] = df["Title"].replace('Mme', 'Mrs')

# ===================== 2. 特征选择与编码 =====================
# 选择最终特征（比之前多了5个强特征）
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
X = df[features]
y = df["Survived"]

# 独热编码分类变量
X = pd.get_dummies(X, columns=["Sex", "Embarked", "Title"], drop_first=True)

# ===================== 3. 模型训练与调参 =====================
# 固定随机种子，保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 调参后的随机森林（比默认模型更稳）
model = RandomForestClassifier(
    n_estimators=200,  # 树的数量，默认100，提升到200更稳定
    max_depth=5,       # 树的最大深度，防止过拟合
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# ===================== 4. 模型评估（多维度） =====================
# 4.1 测试集准确率
y_pred = model.predict(X_test)
print(f"🎯 测试集准确率：{accuracy_score(y_test, y_pred):.2%}")

# 4.2 5折交叉验证（更客观的评估）
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"📊 5折交叉验证平均准确率：{cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

# 4.3 分类报告（精确率、召回率、F1值）
print("\n【分类报告】")
print(classification_report(y_test, y_pred))

# ===================== 5. 特征重要性可视化 =====================
importances = model.feature_importances_
features = X.columns

# 按重要性排序
indices = np.argsort(importances)[::-1]
print("\n【特征重要性（Top10）】")
for i in indices[:10]:
    print(f"{features[i]}: {importances[i]:.2%}")

# 画图
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances[:10])), importances[indices[:10]], align='center')
plt.xticks(range(len(importances[:10])), features[indices[:10]], rotation=45)
plt.title("Top10 特征对生还的影响程度")
plt.ylabel("重要性")
plt.tight_layout()
plt.show()

# ===================== 6. 单样本预测（无警告版） =====================
def predict_survival(pclass, sex, age, fare, embarked, family_size, title):
    # 构造和训练时完全一致的DataFrame
    data = {
        "Pclass": [pclass],
        "Age": [age],
        "Fare": [fare],
        "FamilySize": [family_size],
        "IsAlone": [1 if family_size == 1 else 0],
        "Sex_male": [1 if sex == "male" else 0],
        "Embarked_Q": [1 if embarked == "Q" else 0],
        "Embarked_S": [1 if embarked == "S" else 0],
        "Title_Miss": [1 if title == "Miss" else 0],
        "Title_Mr": [1 if title == "Mr" else 0],
        "Title_Mrs": [1 if title == "Mrs" else 0],
        "Title_Rare": [1 if title == "Rare" else 0]
    }
    df_person = pd.DataFrame(data)
    # 补全所有缺失列（和训练集列名完全一致）
    for col in X.columns:
        if col not in df_person.columns:
            df_person[col] = 0
    result = model.predict(df_person)
    return "生还" if result[0] == 1 else "死亡"

# 测试预测
print("\n【单样本预测测试】")
print("30岁男性，1等舱，票价100，S港登船，独自出行，称谓Mr：", 
      predict_survival(1, "male", 30, 100, "S", 1, "Mr"))
print("25岁女性，3等舱，票价20，S港登船，家庭2人，称谓Miss：", 
      predict_survival(3, "female", 25, 20, "S", 2, "Miss"))

print("\n🎉 泰坦尼克号项目完整收尾！")