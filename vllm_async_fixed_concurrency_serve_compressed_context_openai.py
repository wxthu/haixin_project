"""
使用 vLLM OpenAI 接口进行异步批量摘要（固定并发度 4）：
- 先在本地 GPU 上用 LLMLingua 对长文本做压缩（在线程池中执行，避免阻塞事件循环）
- 再以固定并发度的方式发送流式推理请求，并记录 TTFT / E2E 等指标
- 始终保持同时有 4 个请求在模型端执行（除非剩余任务不足 4 个）
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

# 固定并发度
CONCURRENCY = 30

# --- 基础配置 ---
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8090/v1"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
INPUT_FILE = "media_long_context.jsonl"
OUTPUT_FILE = f"summary_results_async_concurrency_{CONCURRENCY}.jsonl"



client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 初始化压缩器 (GPU 模式)
llm_lingua = PromptCompressor(
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True
)


def set_seed(seed=42):
    """固定随机种子，保证复现性（这里只影响与 random / numpy 相关的部分）"""
    random.seed(seed)
    np.random.seed(seed)
    # 如有 torch，可一并固定
    # torch.manual_seed(seed)


def segment_and_compress(content: str) -> str:
    """
    简单的压缩逻辑（同步版本，在线程池中调用）

    思路：
    - 只对「####相关文档信息列表[...]」这一段做压缩，保留其它结构不变
    - 压缩使用 LLMLingua 的 longllmlingua 排序方法
    """
    pattern = r'(####相关文档信息列表[:：,，]\s*\[.*?\])'
    parts = re.split(pattern, content, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if part.startswith("####相关文档信息列表"):
            res = llm_lingua.compress_prompt(
                part,
                rate=0.5,
                rank_method="longllmlingua"
            )
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
        # 1. 压缩（在线程池中执行）
        raw_content = data.get("content", "")
        input_length = len(raw_content)
        # compressed_prompt = await segment_and_compress_async(raw_content)
        # 如需对比，可直接用原文：
        compressed_prompt = raw_content

        # 2. 准备消息
        messages = [
            {"role": "system", "content": "只输出“####总结内容”。"},
            {
                "role": "user",
                "content": compressed_prompt + "\n\n请只输出####总结内容："
            },
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
            extra_body={"enable_thinking": True},
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
                "e2e": round(e2e, 4),
            },
            "summary": "".join(full_content).strip(),
        }

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pbar.update(1)


async def worker(name: int, queue: asyncio.Queue, pbar: tqdm, start_time: float):
    """
    固定并发 worker：

    - 每个 worker 不断从队列中取出一个样本
    - 记录当前时间作为 arrival_abs_time
    - 调用 handle_request
    - 队列为空并且所有任务 done 后自动退出
    """
    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            break

        arrival_abs_time = time.perf_counter()
        try:
            await handle_request(item, arrival_abs_time, pbar, start_time)
        finally:
            queue.task_done()


async def process_data():
    """
    入口协程（固定并发度版本）：

    - 一次性读取所有输入样本到内存
    - 把样本放入 asyncio.Queue
    - 启动 CONCURRENCY 个 worker 并发消费队列
    - 始终保持最多 CONCURRENCY 个请求在飞（除非剩余不足）
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_data)} items. Running with fixed concurrency={CONCURRENCY} ...")
    pbar = tqdm(total=len(all_data))
    start_time = time.perf_counter()

    queue: asyncio.Queue = asyncio.Queue()
    for item in all_data:
        await queue.put(item)

    workers = [
        asyncio.create_task(worker(i, queue, pbar, start_time))
        for i in range(CONCURRENCY)
    ]

    await queue.join()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    pbar.close()


if __name__ == "__main__":
    set_seed(42)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass
    asyncio.run(process_data())