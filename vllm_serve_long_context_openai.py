
# python -m vllm.entrypoints.api_server --model Qwen/Qwen3-32B-AWQ --tensor-parallel-size 1 --trust-remote-code --max-model-len 32768 --port 8090
# VLLM_ATTENTION_BACKEND=FLASHINFER python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B-AWQ --tensor-parallel-size 1 --trust-remote-code --max-model-len 32768 --port 8090
# --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5}'
# python -m vllm.entrypoints.openai.api_server  --model Qwen/Qwen3-32B-AWQ  --tensor-parallel-size 1 --trust-remote-code --max-model-len 131072  --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'   --port 8099

import json
import time
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
OUTPUT_FILE = "summary_results_under_raw_inputs.jsonl"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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
    # llm_lingua = PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)

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
                
                # # 使用 LLMLingua 进行提示压缩
                # compressed_prompt = llm_lingua.compress_prompt(
                #     raw_content,
                #     question="question", 
                #     rate=0.50,
                #     condition_in_question="after_condition",
                #     reorder_context="sort",
                #     dynamic_context_compression_ratio=0.0,
                #     condition_compare=True,
                #     context_budget="+100",
                #     rank_method="longllmlingua",
                # )
                # raw_content = compressed_prompt["compressed_prompt"]
                
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
                    temperature=0,
                    max_tokens=1024,
                    stream=True,
                    extra_body={"enable_thinking": True}
                )
                
                # for chunk in response:
                #     # 捕获第一个有效 Token 的时间            
                #     if ttft is None and chunk.choices[0].delta.content:
                #         ttft = time.perf_counter() - start_time
                    
                #     content_piece = chunk.choices[0].delta.content
                #     if content_piece:
                #         full_content.append(content_piece)
                for chunk in response:
                    delta = chunk.choices[0].delta
                    # 只要有任何文本输出（无论是思考还是正式回答），就记录首字时间
                    has_content = getattr(delta, "content", None)
                    has_reasoning = getattr(delta, "reasoning_content", None) # 适配思考链
                    
                    if ttft is None and (has_content or has_reasoning):
                        ttft = time.perf_counter() - start_time
                    
                    # 拼接内容
                    content_piece = has_content or ""
                    full_content.append(content_piece)
                # --- 计时终点 ---
                end_to_end_time = time.perf_counter() - start_time
                
                # 3. 结果清洗
                summary_result = "".join(full_content).replace("####总结内容：", "").strip()

                # 4. 写入结果
                result_item = {
                    "metrics": {
                        "input_char_len": input_length,
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

    print(f"\n任务完成！结果文件：{OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
 