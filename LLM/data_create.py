# 模型加载
# 模型prompt
# 模型数据生成
# 数据保存
# 参考
# https://github.com/datawhalechina/self-llm/blob/master/examples/Tianji-%E5%A4%A9%E6%9C%BA/readme.md
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# from peft import PeftModel
import time
import json
import random
import datetime
import re

mode_path = 'C:/jhq/huggingface_model/LLM-Research/Meta-Llama-3___1-8B-Instruct'
# mode_path = 'C:/jhq/huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# lora_path = './output/llama3_1_instruct_lora/checkpoint-20' # 这里改称你的 lora 输出对应 checkpoint 地址

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    mode_path, quantization_config=nf4_config)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path,
                                             quantization_config=nf4_config,
                                             # 設定使用的設備，此處指定為 GPU 0
                                             device_map={'': 0}
                                             ).eval()

# 加载lora权重
# model = PeftModel.from_pretrained(model, model_id=lora_path)

# 可利用大模型补充不同对象  当前28种
name_list = ['赵老师', '大舅', '大伯', '李总', '邻居赵大妈', '母亲', '姐姐', '妹妹', '哥哥', '弟弟', '爷爷', '奶奶', '外公',
             '外婆', '伯母', '叔叔', '阿姨', '堂兄', '堂妹', '表哥', '表妹', '导师', '同学', '同事', '领导',
             '邻居', '老板', '医生', ]

# 可利用大模型补充对应场景 当前18种
scenes = ['生日', '春节', '元宵节', '端午节', '七夕节', '中秋节',
          '重阳节', '除夕', '腊八节', '谈判顺利', '乔迁新居', '周年纪念', '新婚快乐', '家庭和睦', '比赛取得好成绩', '发财', '工作升职 ', '康复', ]

# 可利用大模型补充不同风格，加入更多 fewshot 造出更好的数据
styles = {
    "小红书":
    {
        "style_temple": "小红书风格，每条加入1-2个emoji表情包来增加趣味性。\n### 注意，你要参考下列句子的艺术风格进行祝福语撰写（注意！只看造句风格），祝福语结尾都带上语气助词词，参考句子为：{} ###",
        "if_example": True,
        "examples":
        [
            '默念你的名,祝你前途云蒸霞蔚，灿若星河。愿你度过的叫吉时，得到的叫如愿！',
            '希望你岁末将至，敬颂冬绥，平安喜乐，万事胜意。',
            '希望你不用奔赴大海，也能看到春暖花开；不用颠沛流离，也能遇到一生所伴！',
            '祝我们好在春夏秋冬,祝你阔谈，祝你烂漫，祝你和自己相约在风里，此后只剩欢愉。',
            '希望你可以明确地爱，直接的厌恶，真诚的喜欢，站在太阳下的坦荡，大声无愧地称赞自己，学会爱自己！',
            '前方荣光万丈，身后温暖一方，凡是过往，皆为序章。',
            '愿所念之人 平安喜乐。愿所想之事 顺心如意！',
        ]
    },
    "正常":
    {
        "style_temple": "正常风格，有礼貌即可",
        "if_example": False,
        "examples": []
    },
    "严肃":
    {
        "style_temple": "商业严肃风格，要求用在职场或长辈祝福上，显得有礼貌、干练,句子可以长一些",
        "if_example": False,
        "examples": []
    }
}

random_finalprompt_sentence = [
    '',  # 默认情况
    '回答中可以不出现对象称谓和场景信息，也不用出现“愿你”“祝你”（对自己的长辈需要出现对象称谓和祝你），',
    '回答中可以不出现对象称谓和场景信息，',
    '回答中不用出现“愿你”“祝你”',
]
final_prompt = """
该祝福语字数小于 {} 字。 \n
请根据对象称谓及场景，写出符合对象的身份和场景气氛的祝福文案。要求的风格是：{} \n，注意不要有标题混在其中，对象称谓是：{}，祝福场景是：{}。 \n
{} 根据不同对象用不同的语气（尊敬、诙谐搞笑、亲近），请直接返回祝福文本，不要说任何其他话：
"""

# 文本分割函数


def split_text(text):
    pattern = re.compile(r'(.*?)</think>(.*)', re.DOTALL)  # 定义正则表达式模式
    match = pattern.search(text)  # 匹配 <think>思考过程</think>回答

    if match:  # 如果匹配到思考过程
        think_content = match.group(1).strip()  # 获取思考过程
        answer_content = match.group(2).strip()  # 获取回答
    else:
        think_content = ""  # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip()  # 直接返回回答

    return think_content, answer_content


if __name__ == "__main__":
    ##### 此处配置 #####
    roop_count = 2
    now_count = 0
    # stylename = "小红书" # 小红书、正常、严肃
    output_number_limit = 50  # 限制回答输出长度，严肃的100，普通的小于20
    ##### 此处配置 #####

    for roop in range(roop_count):
        conversations = []
        for name in name_list:
            for scene in scenes:
                for stylename in styles.keys():
                    try:
                        if styles[stylename]['if_example']:
                            style_prompt = styles[stylename]['style_temple'].format(
                                random.choice(styles[stylename]['examples']))
                        else:
                            style_prompt = styles[stylename]['style_temple']
                        input_prompt = final_prompt.format(
                            output_number_limit, style_prompt, name, scene, random.choice(random_finalprompt_sentence))
                        messages = [
                            {"role": "system",
                                "content": "你现在是一个精通言语表达、热爱他人、尊重长辈、富有文采的送祝福大师，请你编辑一条文本，表示对应场景的祝福语"},
                            {"role": "user", "content": input_prompt, "temperature": 1}
                        ]
                        # input_prompt = "春节到了，帮我写一下祝福语给王老师，不少于50字，不要说其他话"
                        # messages = [
                        #         {"role": "user", "content": input_prompt}
                        # ]

                        input_ids = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)

                        model_inputs = tokenizer(
                            [input_ids], return_tensors="pt").to(model.device)
                        # deepseek-r1-7b有思考过程，给太小结果生成不出来
                        generated_ids = model.generate(
                            model_inputs.input_ids, max_new_tokens=8192)
                        # generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512) # Llama-3，512就够了
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        response = tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True)[0]
                        # response = get_data_ds(input_prompt)
                        now_count += 1

                        # if '\n' in str(response):
                        #     response = str(response).split('\n')[0]
                        think_content, answer_content = split_text(
                            response)  # 调用split_text函数，分割思考过程和回答

                        print(name, scene, think_content,
                              'response:', answer_content)
                        print("当前生成数目：", now_count)
                        if stylename == '正常':
                            # 默认不加风格指定
                            _input_prompt = f"祝{name}{scene}"
                        else:
                            _input_prompt = f"祝{name}{scene},{stylename}风格"
                        print("input:", _input_prompt)

                        conversation = {
                            "conversation": [
                                {
                                    "system": "你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福",
                                    "src_input": input_prompt,
                                    "style_name": stylename,
                                    "input": _input_prompt,
                                    "output": str(answer_content).replace('\"', '')
                                }
                            ]
                        }

                        # 将对话加入到列表中
                        conversations.append(conversation)
                    except Exception as e:
                        print(e)
                        continue

        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_path = f"./run/wishes_{now_time}.json"
        with open(file_path, "w", encoding='utf8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)
