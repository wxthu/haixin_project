import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# 1. 加载模型
model_name = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_len = tokenizer.model_max_length
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
model.eval()

def get_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        # 按照 Qwen-Embedding 惯例，取最后一个 token 的隐藏状态
        embeddings = outputs.last_hidden_state[:, -1, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def load_summaries(file_path):
    """从 jsonl 文件中提取 summary 字段"""
    summaries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 提取 summary，去掉 <think> 标签部分以获得纯净的输出文本
            content = data.get("summary", "")
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()
            summaries.append(content)
    return summaries

# 2. 文件路径
file_raw = "summary_results_under_raw_inputs.jsonl"
file_comp = "summary_results_under_compressed_inputs.jsonl"

# 3. 读取内容
raw_texts = load_summaries(file_raw)
comp_texts = load_summaries(file_comp)

# 确保两个文件行数一致
min_len = min(len(raw_texts), len(comp_texts))

print(f"开始计算相似度，共 {min_len} 对数据...\n")
print(f"{'索引':<5} | {'原始总结预览':<20} | {'压缩总结预览':<20} | {'相似度':<8}")
print("-" * 75)

# 4. 逐对计算并输出
for i in range(min_len):
    t1 = raw_texts[i]
    t2 = comp_texts[i]
    
    # 获取向量
    embeddings = get_embedding([t1, t2])
    # 计算余弦相似度
    similarity = torch.matmul(embeddings[0], embeddings[1].T).item()
    
    # 打印结果（截断预览文本以便查看）
    # print(f"{i:<5} | {t1[:18]:<20} | {t2[:18]:<20} | {similarity:.4f}")
    print(f"{i:<5} || score ---> {similarity:.4f}")
    print(f"Raw Output ==> {t1:<20} ")    
    print(f"Compressed ==> {t2:<20} ")