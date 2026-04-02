'''
Description  : 
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-02
LastEditors  : linjie
LastEditTime : 2026-04-02
'''
# ========================
# 1. 导入需要的库
# ========================
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

# ========================
# 2. 加载数据（鸢尾花）
# ========================
iris = load_iris()
X = iris.data      # 特征：花瓣、花萼的长宽
y = iris.target    # 标签：花的种类(0/1/2)

# ========================
# 3. 定义要搜索的参数范围
# ========================
# 字典格式：key是参数名，value是要试的列表
param_grid = {
    'max_depth': [2, 3, 4, 5],              # 树的最大深度
    'n_estimators': [50, 100, 150],         # 森林里有多少棵树
    'min_samples_split': [2, 5]              # 节点最少多少样本才分裂
}

# ========================
# 4. 创建基础模型 + 网格搜索对象
# ========================
rf = RandomForestClassifier(random_state=42)

# GridSearchCV说明：
# estimator：用哪个模型
# param_grid：要搜的参数
# cv=5：5折交叉验证（更稳）
# scoring='accuracy'：用准确率评分
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # 用满CPU，加速搜索
)

# 开始搜索！
grid_search.fit(X, y)

# ========================
# 5. 输出搜索结果
# ========================
print("="*60)
print("🔍 网格搜索最优结果")
print("="*60)

# 最好的准确率
print("最优交叉验证准确率：", grid_search.best_score_)

# 最优的一组参数
print("最优参数组合：", grid_search.best_params_)

# 直接拿到已经用最优参数训练好的模型
best_model = grid_search.best_estimator_

# ========================
# 6. 用最优模型预测
# ========================
sample = [[5.1, 3.5, 1.4, 0.2]]  # 随便一朵花
pred = best_model.predict(sample)
print("\n预测类别：", iris.target_names[pred[0]])