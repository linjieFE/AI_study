'''
Description  : 
Version      : 
Company      : Flexiv-SH
Author       : Linjie Zheng
Date         : 2026-03-31
LastEditors  : linjie
LastEditTime : 2026-03-31
'''
# 写入文件
'''
 def open(
    file: FileDescriptorOrPath,
    mode: OpenTextMode = "r" | "w" | "a" | "x" | "b" | "t" | "u" | "r+" | "w+" | "a+" | "x+" | "b+" | "t+" | "u+",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: _Opener | None = None
    ) -> _TextIO[Any]: ...
'''
with open('text.txt',"w",encoding="utf-8") as f:
    f.write("Hello, AI\n!")
    f.write("今天学习文件读写\n!")

# 读取文件
with open('text.txt',"r",encoding="utf-8") as f:
    content = f.read()
    print(content)

import json

# 字典 
user = {
    "name": "张三",
    "age": 25,
    "job": "前端工程师"
}
# 写入
with open('user.json',"w",encoding="utf-8") as f:
    json.dump(user,f,ensure_ascii=False,indent=4),

# 读取
with open('user.json',"r",encoding="utf-8") as f:
    data = json.load(f)
    print(f'data["name"]=>{data["name"]}')
    print(f'data["job"]=>{data["job"]}')
# 转换为 JSON
json_data = json.dumps(user)
print(json_data)

# 转换为字典
user = json.loads(json_data)
print(f'转换为字典=>{user}')

#字典 
students = {
    "name": "张三",
    "age": 25,
}
#写入
with open('students.json',"w",encoding="utf-8") as f:
    json.dump(students,f,ensure_ascii=False,indent=4)

json_data = json.dumps(students)
print(json_data)

print(json.loads(json_data))