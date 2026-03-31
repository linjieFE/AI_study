'''
Description  : 90天学习计划 - 第一天-字典（键值对）
Version      : v1.0
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-30
LastEditors  : linjie
LastEditTime : 2026-03-30
'''
# 字典（键值对）
user = {
    "name": "小明",
    "age": 25,
    "job": "前端工程师"
}

# 访问
print(user["name"])
print(user["age"])

# 修改
user["age"] = 26
print(user)

# 新增
user["city"] = "上海"
print(user)

skillList = [
    "Python","PyTorch"
    # {"skillName": "Python", "skillLevel": "中级", "skillScore": 80},
    # {"skillName": "PyTorch", "skillLevel": "初级", "skillScore": 70},
    # {"skillName": "AI", "skillLevel": "高级", "skillScore": 90},
]
skillList.append("AI")
skillList.remove("Python")
skillList.pop()
print(skillList)
profile = {
    "name":"码力全开",
    "age":25,
    "skill":skillList
}
print(profile)