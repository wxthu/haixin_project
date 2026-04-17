
# python -m vllm.entrypoints.api_server --model Qwen/Qwen3-32B-AWQ --tensor-parallel-size 1 --trust-remote-code --max-model-len 32768 --port 8090
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B-AWQ --tensor-parallel-size 1 --trust-remote-code --max-model-len 32768 --port 8090 --no-enable-prefix-caching

import json
import time
import re
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm
import torch
from llmlingua import PromptCompressor


# --- 配置参数 ---
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8090/v1"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
INPUT_FILE = "media_long_context.jsonl"
OUTPUT_FILE = "summary_results_under_compressed_inputs_elbow.jsonl"


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


import json
import re

def segment_context(content: str):
    # 核心修正：
    # 1. [:,，] 兼容中文冒号、英文冒号、英文逗号和中文逗号
    # 2. \s* 允许中间有空格
    # 3. re.DOTALL 允许 .*? 匹配跨行内容
    pattern = r'(####相关文档信息列表[:：,，]\s*\[.*?\])'
    
    # 使用 re.split 拆分，括号会保留匹配项
    # flags=re.DOTALL 是关键，处理内容里的换行符
    raw_parts = re.split(pattern, content, flags=re.DOTALL)
    
    # 清理：去首尾空格，去掉空字符串
    segments = [p.strip() for p in raw_parts if p and p.strip()]
    
    return segments


def build_messages(content: str) -> List[Dict[str, str]]:
    """
    在这里进行预处理（如清洗、截断等）
    """
    # 模拟预处理逻辑...
    system = (
        "你将接收一段包含任务说明与输入信息的文本。"
        "请严格按照该文本中的规则生成结果，并且只输出最终的“####总结内容”。"
    )
    user = content.strip() + "\n\n请只输出####总结内容："
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def process_data():
    line_count = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8'))
    print(f"开始处理: {INPUT_FILE}，总计 {line_count} 条数据...")

    # 注意：LLMLingua 加载也需要时间，这里不计入推理测速
    llm_lingua = PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)
    cr = []  # 用于记录压缩率

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=line_count, desc="推理中"):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                raw_content = data.get("content", "")
                input_length = len(raw_content)

                # --- 计时起点：从预处理开始 ---
                start_time = time.perf_counter()
                segmented_contents = segment_context(raw_content)
                for i, seg in enumerate(segmented_contents):
                    if seg.startswith("####相关文档信息列表"):
                
                        # 使用 LLMLingua 进行提示压缩
                        compressed_prompt = llm_lingua.compress_prompt(
                            seg,
                            question="question", 
                            rate=0.5,
                            adaptive_compression=True,
                            adaptive_strategy="elbow",
                            # adaptive_mass_alpha=0.8,
                            condition_in_question="after_condition",
                            reorder_context="sort",
                            dynamic_context_compression_ratio=0.0,
                            condition_compare=True,
                            context_budget="+100",
                            rank_method="longllmlingua",
                        )
                        segmented_contents[i] = compressed_prompt["compressed_prompt"]
                        cr.append(compressed_prompt["compression_rate"])

                raw_content = "".join(segmented_contents)
                compressed_time = time.perf_counter() - start_time

                # 1. 执行预处理逻辑
                messages = build_messages(raw_content)
                
                # --- 统计初始化 ---
                ttft = None
                full_content = []

                # 2. 发送请求
                # 提示：Qwen3 32B 在 AWQ 量化下，首字响应受 Prompt 长度影响较大
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                    stream=True,
                    extra_body={"enable_thinking": True}
                )
                
                for chunk in response:
                    # 捕获第一个有效 Token 的时间
                    if ttft is None and chunk.choices[0].delta.content:
                        ttft = time.perf_counter() - start_time
                    
                    content_piece = chunk.choices[0].delta.content
                    if content_piece:
                        full_content.append(content_piece)
                
                # --- 计时终点 ---
                end_to_end_time = time.perf_counter() - start_time
                
                # 3. 结果清洗
                summary_result = "".join(full_content).replace("####总结内容：", "").strip()

                # 4. 写入结果
                result_item = {
                    "metrics": {
                        "input_char_len": input_length,
                        "compressed_time_seconds": round(compressed_time, 4),
                        "ttft_seconds": round(ttft, 4) if ttft else None,
                        "e2e_seconds": round(end_to_end_time, 4),
                        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "summary": summary_result
                }
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()
                
            except Exception as e:
                print(f"\n[Error] 处理单条数据失败: {e}")
                continue
        
        print(f"平均压缩率: {sum(cr)/len(cr) if cr else 0:.4f}")


if __name__ == "__main__":
    process_data()