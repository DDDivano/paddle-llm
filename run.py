#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
run
"""
import os

# 指定要搜索的目录和前缀
folder_path = "./"  # 目录路径
prefix = "test_"    # 文件名前缀

# 获取目录下所有文件名
file_names = os.listdir(folder_path)

# 遍历文件名，找到以指定前缀开头的文件名
matched_files = []
for file_name in file_names:
    if file_name.startswith(prefix):
        matched_files.append(file_name)

print(matched_files)

for case in matched_files:
    os.system("python3.9 -m pytest {} --alluredir=./report".format(case))

