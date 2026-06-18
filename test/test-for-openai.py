from openai import OpenAI

client = OpenAI(
    api_key = "EMPTY",
    base_url = ""
)

# 微调模型调用
response = client.chat.completions.create(
    model = "qwen-lora",
    messages = [{
        "role":"user",
        "content":"从以下政企文本中提取公司全称、注册资本、法定代表人、成立日期、经营范围、主体类型、风险等级、资产规模、是否获得政策支持。以JSON格式输出。\n重庆绿野农业科技有限公司成立于2019年，注册资本2000万元，法定代表人张明。"
    }],
    temperature=0
)

print(response.choices[0].message.content)