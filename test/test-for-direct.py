import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "/data1/lora_finetune/models/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="cuda:7"
)

tokenizers = AutoTokenizer.from_pretrained(
    "/data1/lora_finetune/models/Qwen2.5-7B"
)

model = PeftModel.from_pretrained(
    base_model,
    "/data1/lora_finetune/output/qwen_lora"
)

instruction = "从以下政企文本中提取公司全称、注册资本、法定代表人、成立日期、经营范围、主体类型、风险等级、资产规模、是否获得政策支持。以JSON格式输出。"
text = f"{instruction}\n重庆绿野农业科技有限公司成立于2019年，注册资本2000万元，法定代表人张明。"

inputs = tokenizers(text, return_tensors="pt").to("cuda:7")
outputs = model.generate(**inputs, max_new_tokens=256,temperature=0)
result = tokenizers.decode(outputs[0], skip_special_tokens=True)

print(result)