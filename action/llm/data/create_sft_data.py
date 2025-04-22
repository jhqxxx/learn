'''
Author: jhq
Date: 2025-04-21 21:55:30
LastEditTime: 2025-04-22 15:25:23
Description: 
'''
import os
import re
import json

def make_sft_data_farming(txt_path):
    file_list = os.listdir(txt_path)
    pattern = re.compile(r"【.*】")
    pattern2 = re.compile(r"（.*）")
    pattern3 = re.compile(r"\s\s*")
    total_line = []
    for i in range(len(file_list)):
        file = file_list[i]
        txt_file = os.path.join(txt_path, file)
        file_name = file.split(".")[0]
        file_name = pattern.sub("", file_name)
        file_name = pattern2.sub("", file_name)
        file_name = file_name.replace("昌都市", "")        
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
        # print(file_name)
        
        file_txt = ""
        with open(txt_file, "r", encoding="utf-8") as f:
            file_txt += f.read()
        file_txt = file_txt.strip().replace("\n", "")        
        file_txt = file_txt.strip().replace("▲", " ")
        file_txt = file_txt.strip().replace("▼", " ")        
        file_txt = file_txt.strip().replace("⭐", "")        
        file_txt = file_txt.strip().replace("·", "")
        file_txt = pattern3.sub(" ", file_txt)
        # if len(file_txt) < 20 or len(file_txt) > 2048:
        if len(file_txt) < 20:
            continue
        data_dict = {
            "instruction": file_name,
            "input": "",
            "output": file_txt,
        }
        data_line = json.dumps(data_dict, ensure_ascii=False) + "\n"
        total_line.append(data_line)
    return total_line
        # print(file_name)        
        # print(file_txt)
        # if i > 100:
        #     break

if __name__ == "__main__":
    # txt_path = r"C:\jhq\rag_file\farming\txt\changdu"
    txt_path = r"C:\jhq\rag_file\farming\txt\zhongguonongye_zhongzhiye"
    jsonl_path = r"C:\jhq\rag_file\farming\sft_data"
    total_line = make_sft_data_farming(txt_path)
    with open(os.path.join(jsonl_path, "nongye_sft_data.json"), "w", encoding="utf-8") as f:
        f.writelines(total_line)