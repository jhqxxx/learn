'''
Author: jhq
Date: 2025-04-21 21:55:30
LastEditTime: 2025-04-26 11:27:07
Description: 
'''
import os
import re
import json
import random
import glob

def format_file_name(file_name):
    pattern = re.compile(r"【.*】")
    pattern2 = re.compile(r"（.*）")    
    file_name = file_name.replace(".", "")
    file_name = pattern.sub("", file_name)
    file_name = pattern2.sub("", file_name)
    file_name = file_name.replace("昌都市", "")        
    file_name = file_name.replace("[", "")
    file_name = file_name.replace("]", "")
    file_name = file_name.replace("固镇县", "")          
    file_name = file_name.replace("蚌埠市", "")  
    file_name = file_name.replace("巴中市", "")
    file_name = file_name.replace("测报方法与防治技术", "")        
    file_name_list = file_name.split("")
    if len(file_name_list) == 2:
        if file_name_list[0] == file_name_list[1]:
            file_name = file_name_list[0]
    file_name = file_name.replace("", "")
    file_name = file_name.replace(" ", "")
    return file_name

def format_file_txt(file_txt):
    pattern3 = re.compile(r"\s\s*")
    file_txt = file_txt.strip().replace("\n", "")        
    file_txt = file_txt.replace("▲", " ")
    file_txt = file_txt.replace("▼", " ")        
    file_txt = file_txt.replace("⭐", "")        
    file_txt = file_txt.replace("·", "")
    file_txt = pattern3.sub(" ", file_txt)
    return file_txt

def make_sft_data_farming(txt_path):
    # file_list = os.listdir(txt_path)
    file_sufix = ".txt"
    pathname = txt_path + "/*/*" + file_sufix
    file_list = glob.glob(pathname=pathname, recursive=True)
    print(len(file_list))
    total_line = []
    name_list = []
    for i in range(int(len(file_list))):
        file = file_list[i]
        (path, file_name) = os.path.split(file)
        file_name = file_name.replace(".txt", "")   # 因为文件都是txt，
        
        file_txt = ""
        with open(file, "r", encoding="utf-8") as f:
            file_txt += f.read()
        if len(file_txt) < 20 or len(file_txt) > 512:
            continue
        if file_name in file_txt:
            file_txt = file_txt.replace(file_name, "")
        file_name = format_file_name(file_name)
        file_txt = format_file_txt(file_txt)
        if file_name in name_list:
            continue
        name_list.append(file_name)
        data_dict = {
            # "instruction": "以下是一个农业技术文章标题或农业问题，请完成文章内容或回答问题。",
            "instruction": file_name,
            "input": "",
            "output": file_txt,
        }
        # data_line = json.dumps(data_dict, ensure_ascii=False) + "\n"
        total_line.append(data_dict)
    return total_line

if __name__ == "__main__":
    # txt_path = r"C:\jhq\rag_file\farming\txt\changdu"
    txt_path = r"C:\jhq\rag_file\farming\txt"
    jsonl_path = r"C:\jhq\rag_file\farming\sft_data"
    total_line = make_sft_data_farming(txt_path)
    print("total_line:", len(total_line))
    with open(os.path.join(jsonl_path, "nongye_sft_data.json"), "w", encoding="utf-8") as f:
        json.dump(total_line, f, indent=2, ensure_ascii=False)
    # with open(os.path.join(jsonl_path, "nongye_sft_data.json"), "w", encoding="utf-8") as f:
    #     f.writelines(total_line)