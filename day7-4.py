'''
Description  : 90天学习计划 - 第7天-机器学习-流水线
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    # 标准化（归一化）
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline               # 流水线核心
from sklearn.metrics import accuracy_score

# ========================
# 2. 数据准备
# ========================
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ========================
# 3. 定义流水线 Pipeline
# ========================
# 格式：列表里放元组 (步骤名称, 对应的处理类)
pipeline = Pipeline([
    # 第一步：对特征做标准化（归一化）
    ("scaler", StandardScaler()),
    # 第二步：放入随机森林模型训练
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
])

# ========================
# 4. 训练：一句话搞定所有步骤
# ========================
pipeline.fit(X_train, y_train)

# ========================
# 5. 预测：同样一句话
# ========================
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("流水线模型准确率：", acc)

# ========================
# 6. 新样本预测
# ========================
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = pipeline.predict(sample)
print("\n预测花的种类：", iris.target_names[pred[0]])

# ========================
# 7. 保存整个流水线（包含预处理+模型）
# ========================
import joblib
joblib.dump(pipeline, "iris_pipeline.pkl")
print("\n✅ 整条流水线已保存！")