import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 配置区 ---
model_path = "Qwen/Qwen3-32B"  # 替换为你本地存放模型的路径
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载模型: {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", # 自动分配显存
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

def clean_text(text):
    """剔除结果中的 <think> 标签内容"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()

def get_judgement(prompt):
    """调用本地 HF 模型生成评判"""
    messages = [
        {"role": "system", "content": "你是一个严厉的影视百科评估专家。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0# 设置低采样温度以保证评判的一致性
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    # 文件路径
    files = {
        "raw": "media_long_context.jsonl",
        "comp": "B.jsonl",
        "raw_res": "A.jsonl"
    }

    results = []

    with open(files["raw"], 'r') as f_raw, open(files["comp"], 'r') as f_c, open(files["raw_res"], 'r') as f_r:
        for i, (l_raw, l_c, l_r) in enumerate(zip(f_raw, f_c, f_r)):
            data_raw = json.loads(l_raw)
            data_c = json.loads(l_c)
            data_r = json.loads(l_r)

            ans_c = clean_text(data_c.get("summary", ""))
            ans_r = clean_text(data_r.get("summary", ""))
            
            intent = data_raw.get('user_intent', '知识意图')
            limit = "68字" if "检索" in intent else "250字"

            eval_prompt = f"""
请对比以下两个模型对问题的回答，判断哪一个更好。

【原始约束】
- 问题：{data_raw.get('user_problem')}
- 意图：{intent}
- 限制：字数需在 {limit} 以内。

【答案 C (来自 B.jsonl 文件)】
{ans_c}

【答案 R (来自 A.jsonl 文件)】
{ans_r}

请基于约束达标情况、信息准确度进行评价。
必须在输出的第一行写明：结论：[C更好] 或 结论：[R更好] 或 结论：[同等]。
"""
            
            judgement = get_judgement(eval_prompt)
            
            # 简单逻辑提取
            choice = "C" if "C更好" in judgement else "R"
            results.append(choice)
            
            print(f"条目 {i+1} | 优胜: {choice}")
            print(f"理由简述: {judgement.splitlines()[0]}")
            print("-" * 20)

    print(f"\n评估完成！")
    print(f"C 胜出次数: {results.count('C')}")
    print(f"R 胜出次数: {results.count('R')}")
    print(f"C 胜率: {results.count('C')/len(results):.2%}")

if __name__ == "__main__":
    main()