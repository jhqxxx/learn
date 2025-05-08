'''
Author: jhq
Date: 2025-05-01 11:16:49
LastEditTime: 2025-05-01 17:42:52
Description: 
'''
from typing import Any
import httpx
import json
from mcp.server.fastmcp import FastMCP
import asyncio
mcp = FastMCP("weather")

# weather api:
# https://weather.cma.cn/api/now/S1003
# https://weather.cma.cn/api/now/53882
# alert api:
# https://weather.cma.cn/api/map/alarm?adcode=65
WEATHER_API_BASE = "https://weather.cma.cn/api/now/"
ALERT_API_BASE = "https://weather.cma.cn/api/map/alarm?adcode="
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
ACCEPT = "application/json, text/javascript, */*; q=0.01"

def get_city_code() -> dict[str, str]:
    with open("province_city_code.json", "r", encoding="utf-8") as f:
        city_code = json.load(f)
    new_city_code = {}
    for province in city_code:
        for key in province:
            for city in province[key]:
                for name, code in city.items():
                    new_city_code[name] = code
    return new_city_code

city_code_dict = get_city_code()
    
async def make_request(url: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": ACCEPT,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

@mcp.tool()
async def get_forecast(city_name: str) -> str:
    """
    获取指定市/县的天气预报
    Args:
        city_name (str):  city name
    """
    city_name = city_name.replace("市", "").replace("县", "")
    if city_name in city_code_dict:
        city_code = city_code_dict[city_name]
        url = f"{WEATHER_API_BASE}{city_code}"
        response = await make_request(url)
        if not response:
            return "url response is None"
        try:
            temperature = response["data"]["now"]
            forecast = f"""
            气温： {temperature["temperature"]}
            降水： {temperature["precipitation"]}
            气压： {temperature["pressure"]}
            湿度： {temperature["humidity"]}
            风标： {temperature["windScale"]}
            风速： {temperature["windSpeed"]}            
            """
            return forecast
        except KeyError:
            return "url has no data" 
    return f"city code dict not have {city_name}"

async def main():
    city_name = "成都"
    city_code = city_code_dict[city_name]
    url = f"{WEATHER_API_BASE}{city_code}"
    response = await make_request(url)
    print(response)

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # asyncio.run(main())