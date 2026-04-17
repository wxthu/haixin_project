
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B-AWQ --tensor-parallel-size 1 --trust-remote-code --max-model-len 32768 --port 8090 --no-enable-prefix-caching

import json
import time
import re
import asyncio
import random
import numpy as np

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from llmlingua import PromptCompressor

# --- 基础配置 ---
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8090/v1"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
INPUT_FILE = "media_long_context.jsonl"
OUTPUT_FILE = "summary_results_async.jsonl"
TARGET_QPS = 4.0

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
# # 初始化压缩器 (GPU 模式)
# llm_lingua = PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # 如果后续涉及 torch 等框架，也可以一并固定
    # torch.manual_seed(seed)

def segment_and_compress(content: str):
    """简单的压缩逻辑"""
    pattern = r'(####相关文档信息列表[:：,，]\s*\[.*?\])'
    parts = re.split(pattern, content, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if part.startswith("####相关文档信息列表"):
            # 直接调用 GPU 压缩
            res = llm_lingua.compress_prompt(part, rate=0.5, rank_method="longllmlingua")
            parts[i] = res["compressed_prompt"]
    return "".join(parts)

async def handle_request(data: dict, arrival_abs_time: float, pbar: tqdm, start_time: float):
    """处理单个请求的异步协程"""
    try:
        # 1. 压缩 (即便在 GPU，这步目前也会阻塞事件循环一小会儿，初版先这样)
        raw_content = data.get("content", "")
        input_length = len(raw_content)
        # compressed_prompt = segment_and_compress(raw_content)
        compressed_prompt = raw_content
        
        # 2. 准备消息
        messages = [
            {"role": "system", "content": "只输出“####总结内容”。"},
            {"role": "user", "content": compressed_prompt + "\n\n请只输出####总结内容："}
        ]

        # 3. 推理并计时
        ttft = None
        full_content = []
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            stream=True,
            extra_body={"enable_thinking": True}
        )
        
        async for chunk in response:
            if ttft is None and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - arrival_abs_time
            content = chunk.choices[0].delta.content
            if content: full_content.append(content)
        
        # 4. 记录数据
        e2e = time.perf_counter() - arrival_abs_time
        result = {
            "metrics": {
                "input_char_len": input_length,
                "arrival_rel": round(arrival_abs_time-start_time, 4),
                "ttft": round(ttft, 4) if ttft else None,
                "e2e": round(e2e, 4)
            },
            "summary": "".join(full_content).strip()
        }
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pbar.update(1)

async def process_data():
    # 1. 一次性读取数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(all_data)} items. Simulating random arrival...")
    pbar = tqdm(total=len(all_data))
    start_time = time.perf_counter()
    tasks = []

    # 2. 模拟随机到达
    for item in all_data:
        # 使用指数分布模拟随机间隔 (1/QPS)
        wait_time = random.expovariate(TARGET_QPS)
        await asyncio.sleep(wait_time)
        
        current_absolute_time = time.perf_counter()
        # 创建任务并丢进后台执行，不等待它完成
        task = asyncio.create_task(handle_request(item, current_absolute_time, pbar, start_time))
        tasks.append(task)

    # 3. 等待所有后台任务收尾
    await asyncio.gather(*tasks)
    pbar.close()

if __name__ == "__main__":
    set_seed(42)
    # 初始化文件
    with open(OUTPUT_FILE, 'w') as f: pass
    asyncio.run(process_data())