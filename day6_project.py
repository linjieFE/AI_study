'''
Description  : 90天学习计划 - 第6天-项目实战
Version      : v1.0 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-04-01
LastEditors  : linjie
LastEditTime : 2026-04-01
'''
import pandas as pd
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 创建数据 =====================
data = {
    "name": ["张三", "李四", "王五", "赵六", "小明", "小红"],
    "age": [23, 25, None, 29, 24, 27],
    "salary": [8000, 15000, 12000, 25000, 18000, 9000],
    "department": ["技术", "产品", "技术", "管理", "产品", "技术"]
}
df = pd.DataFrame(data)

print("===== 原始数据 =====")
print(df.to_string(index=False))

# ===================== 2. 清洗数据 =====================
df_clean = df.fillna({"age": 25})  # 填充空年龄
print("\n===== 清洗后数据 =====")
print(df_clean.to_string(index=False))

# ===================== 3. 筛选数据 =====================
high_salary = df_clean[df_clean["salary"] > 10000]
print("\n===== 工资 > 10000 的员工 =====")
print(high_salary.to_string(index=False))

# ===================== 4. 统计 =====================
print("\n===== 统计信息 =====")
print("平均工资：", df_clean["salary"].mean())
print("最高工资：", df_clean["salary"].max())

# ===================== 5. 画图 =====================
dept_count = df_clean["department"].value_counts()
plt.bar(dept_count.index, dept_count.values)
plt.title("部门人数统计")
plt.xlabel("部门")
plt.ylabel("人数")
plt.show()

# ===================== 6. 保存结果 =====================
try:
    high_salary.to_json("high_salary.json", force_ascii=False, indent=4)
    print("\n✅ 结果已保存到 high_salary.json")
except Exception as e:
    print("❌ 保存失败：", e)

print("\n🎉 项目运行完成！")