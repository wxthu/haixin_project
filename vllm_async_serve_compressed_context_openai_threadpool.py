"""
使用 vLLM OpenAI 接口进行异步批量摘要：
- 先在本地 GPU 上用 LLMLingua 对长文本做压缩（在线程池中执行，避免阻塞事件循环）
- 再以目标 QPS 随机到达的方式发送流式推理请求，并记录 TTFT / E2E 等指标
"""

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
OUTPUT_FILE = "summary_results_async_threadpool.jsonl"
TARGET_QPS = 32.0  # 目标平均 QPS，用于控制随机到达间隔

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 初始化压缩器 (GPU 模式)，建议在脚本启动时只初始化一次，避免重复加载模型
llm_lingua = PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)


def set_seed(seed=42):
    """固定随机种子，保证不同运行之间的到达分布可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    # 如果后续涉及 torch 等框架，也可以一并固定
    # torch.manual_seed(seed)


def segment_and_compress(content: str):
    """
    简单的压缩逻辑（同步版本，在线程池中调用）

    思路：
    - 只对「####相关文档信息列表[...]」这一段做压缩，保留其它结构不变
    - 压缩使用 LLMLingua 的 longllmlingua 排序方法
    """
    # 用正则把「相关文档信息列表」从全文中切分出来，仅对这一段做压缩
    pattern = r'(####相关文档信息列表[:：,，]\s*\[.*?\])'
    parts = re.split(pattern, content, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if part.startswith("####相关文档信息列表"):
            # 直接调用 GPU 压缩
            res = llm_lingua.compress_prompt(part, rate=0.5, rank_method="longllmlingua")
            parts[i] = res["compressed_prompt"]
    return "".join(parts)


async def segment_and_compress_async(content: str) -> str:
    """
    在线程池中执行压缩，避免阻塞事件循环。
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, segment_and_compress, content)


async def handle_request(data: dict, arrival_abs_time: float, pbar: tqdm, start_time: float):
    """
    处理单个请求的异步协程（压缩放到线程池里）

    - 先在线程池中对原始内容做压缩
    - 再把压缩后的内容送入大模型做流式推理
    - 记录输入长度、TTFT、E2E 等指标并写入结果文件
    """
    try:
        # 1. 压缩（在线程池中执行，避免阻塞事件循环）
        raw_content = data.get("content", "")
        input_length = len(raw_content)
        compressed_prompt = await segment_and_compress_async(raw_content)
        # compressed_prompt = raw_content

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
            if content:
                full_content.append(content)

        # 4. 记录数据
        e2e = time.perf_counter() - arrival_abs_time
        result = {
            "metrics": {
                "input_char_len": input_length,
                "arrival_rel": round(arrival_abs_time - start_time, 4),
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
    """
    入口协程：
    - 一次性读取所有输入样本
    - 按 TARGET_QPS 采样指数分布间隔，模拟随机到达
    - 为每个样本创建异步请求任务并收集结果
    """
    # 1. 一次性读取数据到内存，如果数据量极大可考虑分块读取
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
    with open(OUTPUT_FILE, 'w') as f:
        pass
    asyncio.run(process_data())

