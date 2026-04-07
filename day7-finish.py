'''
Description  : 90天学习计划 - 第7天-机器学习-完整机器学习流水线
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-03
LastEditors  : linjie
LastEditTime : 2026-04-03
'''
# ======================================================
# 项目：鸢尾花分类完整机器学习流水线（无警告版）
# 内容：数据处理 + Pipeline + 调参 + 评估 + 模型保存
# ======================================================

import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# 2. 加载数据
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 3. 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 构建机器学习流水线
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42))
])

# 5. 定义网格搜索参数
param_grid = {
    "rf__n_estimators": [50, 100, 150],
    "rf__max_depth": [3, 4, 5],
    "rf__min_samples_split": [2, 5]
}

# 6. 网格搜索 + 交叉验证
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

# 7. 输出最优结果
print("=" * 60)
print("最优参数：", grid.best_params_)
print("最优交叉验证准确率：", grid.best_score_)
print("=" * 60)

# 8. 测试集评估
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n测试集准确率：", accuracy_score(y_test, y_pred))
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# 9. 保存模型
joblib.dump(best_model, "iris_classifier.pkl")
print("\n✅ 模型已保存：iris_classifier.pkl")

# ==================== 关键修复：预测时用 DataFrame 而不是列表 ====================
loaded_model = joblib.load("iris_classifier.pkl")

# 原来的写法（会报警告）
# sample = [[5.1, 3.5, 1.4, 0.2]]

# 修复后的写法（不报警告）
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names  # 必须带上列名！
)

pred = loaded_model.predict(sample)
print("\n新样本预测结果：", iris.target_names[pred[0]])

print("\n🎉 第三阶段机器学习全部完成！")