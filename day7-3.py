'''
Description  : 90天学习计划 - 第7天-机器学习-模型保存和加载
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-03
LastEditors  : linjie
LastEditTime : 2026-04-03
'''
# ========================
# 1. 导入需要的库
# ========================
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 用于模型保存和加载的库
import joblib

# ========================
# 2. 准备数据
# ========================
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ========================
# 3. 训练模型
# ========================
model = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 测试一下准确率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("模型准确率：", acc)

# ========================
# 4. 保存模型到文件
# ========================
# 保存为 .pkl 或 .joblib 格式
joblib.dump(model, "iris_model.pkl")
print("\n✅ 模型已保存为：iris_model.pkl")

# ========================
# 5. 加载模型
# ========================
# 从文件加载模型
loaded_model = joblib.load("iris_model.pkl")
print("✅ 模型已从文件加载")

# ========================
# 6. 使用加载后的模型预测
# ========================
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = loaded_model.predict(sample)
print("\n预测结果：", iris.target_names[pred[0]])

# 再次验证准确率一致
y_pred2 = loaded_model.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print("加载后模型准确率：", acc2)