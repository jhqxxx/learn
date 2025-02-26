'''
Author: jhq
Date: 2025-02-26 11:44:24
LastEditTime: 2025-02-26 12:49:19
Description: 将生成的多个json文本数据合并
 参考
 https://github.com/datawhalechina/self-llm/blob/master/examples/Tianji-%E5%A4%A9%E6%9C%BA/readme.md
'''
import os
import json


def extract_and_merge_conversations(folder_path, output_file):
    all_conversations = []

    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # 打开并读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 提取需要的字段
                for item in data:
                    for conversation in item['conversation']:
                        extracted = {
                            'system': conversation['system'],
                            'input': conversation['input'],
                            'output': conversation['output']
                        }
                        # 将每个对话包装在一个 'conversation' 键中，并作为独立对象加入列表
                        all_conversations.append({'conversation': [extracted]})

    # 将合并后的所有对话数据写入一个新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(all_conversations, file, ensure_ascii=False, indent=4)


# 使用示例
folder_path = './run'  # 要扫描的文件夹路径
output_file = 'tianji-wishes-chinese-v0.1.json'     # 输出文件的名称和路径
extract_and_merge_conversations(folder_path, output_file)
