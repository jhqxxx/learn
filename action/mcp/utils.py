'''
Author: jhq
Date: 2025-05-01 12:15:58
LastEditTime: 2025-05-01 13:21:51
Description: 
'''
import requests
import json

province_data = 'ABJ,北京|ATJ,天津|AHE,河北|ASX,山西|ANM,内蒙古|ALN,辽宁|AJL,吉林|AHL,黑龙江|ASH,上海|AJS,江苏|AZJ,浙江|AAH,安徽|AFJ,福建|AJX,江西|ASD,山东|AHA,河南|AHB,湖北|AHN,湖南|AGD,广东|AGX,广西|AHI,海南|ACQ,重庆|ASC,四川|AGZ,贵州|AYN,云 南|AXZ,西藏|ASN,陕西|AGS,甘肃|AQH,青海|ANX,宁夏|AXJ,新疆|AXG,香港|AAM,澳门|ATW,台湾'
def get_city_code():
    province_url = f"https://weather.cma.cn/api/dict/province"
    province_data = requests.get(province_url).json()
    province_list = province_data["data"].split("|")
    city_dict = {}
    for province in province_list:
        province_code, province_name = province.split(",")
        city_url = f"https://weather.cma.cn/api/dict/province/{province_code}"
        city_data = requests.get(city_url).json()
        print(city_data)
        city_dict = []
        city_list = city_data["data"].split("|")        
        for city in city_list:
            city_code, city_name = city.split(",")
            city_dict[city_name] = city_code
    with open("city_code.json", "w", encoding="utf-8") as f:
        json.dump(city_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    get_city_code()

