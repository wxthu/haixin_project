#
# python profile_vllm_decode.py --model-path Qwen/Qwen3-8B  --max-new-tokens 200 --num-runs 15
# Qwen/Qwen3-30B-A3B
import os
import argparse
from typing import Dict, List

import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams
from vllm.version import __version__ as vllm_version


def _classify_by_stack_or_key(stack: List[str], key: str) -> str:
    """
    根据调用栈或 key 将 CUDA 事件归为 attention / ffn / other。
    规则基于常见的 self_attn / mlp / gate_proj / up_proj / down_proj / silu / gelu 等命名。
    """
    full = " ".join(stack).lower() if stack else ""
    k = (full or (key or "")).lower()

    if (
        "flash_fwd" in k
        or "attention" in k
        or "softmax" in k
    ):
        return "attention"

    if any(
        p in k
        for p in [
            "gemm",
            "matmul",
            "aten::mm",
            "gemv",
            "linear",
        ]
    ):
        return "linear"

    if any(
        p in k
        for p in [
            "cudadrivergetversion",
            "cudaeventrecord",
            "cudagraphlaunch",
            "cudastreamiscapturing",
            "memcpy",
            "memset",
        ]
    ):
        return "cpu"

    return "other"


def _get_cuda_time(evt) -> float:
    return (
        getattr(evt, "device_time_total", None)
        or getattr(evt, "cuda_time_total", None)
        or getattr(evt, "self_device_time_total", None)
        or 0.0
    )


def _profile_once(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> Dict[str, float]:
    """
    对一次 llm.generate 做 CUDA profiling，返回 attention/ffn/other/us 总和和 total_ms。
    """
    cuda_attention_us = 0.0
    cuda_ffn_us = 0.0
    cuda_other_us = 0.0

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        with_modules=True,
        profile_memory=False,
    ) as prof:
        _ = llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()
    # prof.export_chrome_trace("trace_vllm_decode_simple.json")

    try:
        events = prof.events()
    except Exception:
        events = []

    for evt in events:
        cuda_time = _get_cuda_time(evt)
        if not cuda_time:
            continue
        stack = getattr(evt, "stack", None) or []
        if isinstance(stack, str):
            stack = [stack]
        key = str(getattr(evt, "key", "") or "")
        cat = _classify_by_stack_or_key(stack, key)
        if cat == "attention":
            cuda_attention_us += cuda_time
        elif cat == "linear":
            cuda_ffn_us += cuda_time
        elif cat == "other":
            cuda_other_us += cuda_time

    total_us = cuda_attention_us + cuda_ffn_us + cuda_other_us
    total_ms = total_us / 1000.0

    return {
        "attn_us": cuda_attention_us,
        "ffn_us": cuda_ffn_us,
        "other_us": cuda_other_us,
        "total_us": total_us,
        "total_ms": total_ms,
    }


def test_qwen3_8b_decode_performance(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 100,
    num_runs: int = 5,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    使用 torch.profiler 对 vLLM 的 generate 做 CUDA profiling，
    按 CUDA kernel 名 + 调用栈将时间粗分为 Attention / FFN / Other。

    注意：这里统计的是一次完整生成（prefill + decode）的 CUDA 时间构成；
    在长 decode 情况下，decode 通常是主耗时部分。
    """
    print(f"当前 vLLM 版本：{vllm_version}")
    print(f"测试模型：{model_path}")
    print(f"max_new_tokens={max_new_tokens}, num_runs={num_runs}, batch_size={batch_size}")

    os.environ.setdefault("VLLM_USE_V1", "0")

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        trust_remote_code=True,
        dtype="auto",
        # 与生产 serving 对齐：prefix caching + CUDA Graph（enforce_eager=True 会禁用 graph）
        enable_prefix_caching=True,
        enforce_eager=False,
    )

    prompts = [prompt] * batch_size

    print("\n=== 模型预热（不进入 profiler）===")
    with torch.no_grad():
        _ = llm.generate(prompts=prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()

    attn_pcts = []
    ffn_pcts = []
    other_pcts = []
    total_ms_list = []

    for run_idx in range(num_runs):
        print(f"\n=== 第 {run_idx + 1}/{num_runs} 轮 CUDA profile ===")
        with torch.no_grad():
            stats = _profile_once(llm, prompts, sampling_params)

        if stats["total_us"] <= 0:
            print("  本轮 profiler 未采集到有效 CUDA 时间，跳过。")
            continue

        attn_pct = 100.0 * stats["attn_us"] / stats["total_us"]
        ffn_pct = 100.0 * stats["ffn_us"] / stats["total_us"]
        other_pct = 100.0 * stats["other_us"] / stats["total_us"]
        total_ms = stats["total_ms"]

        attn_pcts.append(attn_pct)
        ffn_pcts.append(ffn_pct)
        other_pcts.append(other_pct)
        total_ms_list.append(total_ms)

        print(
            f"  Attention {attn_pct:.2f}%  Linear {ffn_pct:.2f}%  Other {other_pct:.2f}%  total {total_ms:.2f} ms"
        )

    if not total_ms_list:
        print("没有成功的 profile 轮次，无法统计。")
        return {
            "attention_ratio": 0.0,
            "ffn_ratio": 0.0,
            "other_ratio": 0.0,
            "total_ms": 0.0,
        }

    avg_attn = float(np.mean(attn_pcts))
    avg_ffn = float(np.mean(ffn_pcts))
    avg_other = float(np.mean(other_pcts))
    avg_total_ms = float(np.mean(total_ms_list))

    print("\n" + "=" * 70)
    print("=== Qwen3-8B 生成阶段 CUDA Time Breakdown (平均) ===")
    print(f"Attention: {avg_attn:6.2f}%")
    print(f"FFN     : {avg_ffn:6.2f}%")
    print(f"Other   : {avg_other:6.2f}%")
    print(f"Total   : {avg_total_ms:8.2f} ms / batch")
    print("=" * 70)

    return {
        "attention_ratio": avg_attn,
        "ffn_ratio": avg_ffn,
        "other_ratio": avg_other,
        "total_ms": avg_total_ms,
    }


def _build_prompt_with_length(base_prompt: str, target_length: int) -> str:
    """
    根据给定的 base_prompt 构造近似长度为 target_length 的输入串（按字符数粗略近似）。
    这里只是为了控制大致的 input_length，用于 profile，对语义无特别要求。
    """
    if not base_prompt:
        base_prompt = "你好，请介绍一下大语言模型。"
    if target_length <= len(base_prompt):
        return base_prompt[:target_length]
    repeat = target_length // len(base_prompt) + 1
    s = (base_prompt * repeat)[:target_length]
    return s


def sweep_decode_performance(
    model_path: str,
    base_prompt: str,
    max_new_tokens: int,
    num_runs: int,
    output_path: str,
) -> None:
    """
    在 batch_size=1~4、input_length=2000~20000（步长 2000）的网格上，
    统计 vLLM generate 的 CUDA 时间构成（Attention / Linear / Other），
    并将平均结果写入到 output_path 文件（CSV 格式）。

    某些 (batch_size, input_length) 组合可能 OOM，会自动跳过。
    """
    print(f"当前 vLLM 版本：{vllm_version}")
    print(f"测试模型：{model_path}")
    print(f"max_new_tokens={max_new_tokens}, num_runs={num_runs}")

    os.environ.setdefault("VLLM_USE_V1", "0")

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        trust_remote_code=True,
        dtype="auto",
        enable_prefix_caching=True,
        enforce_eager=False,
    )

    # 预热一次，避免第一轮 profile 偏差太大
    warmup_prompt = _build_prompt_with_length(base_prompt, 2000)
    warmup_prompts = [warmup_prompt]
    print("\n=== 模型预热（不进入 profiler）===")
    with torch.no_grad():
        _ = llm.generate(prompts=warmup_prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()

    batch_sizes = [1, 2, 3, 4]
    input_lengths = list(range(2000, 20001, 2000))

    print(f"\n将结果写入文件：{output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "batch_size,input_length,attention_ratio,ffn_ratio,other_ratio,total_ms\n"
        )

        for batch_size in batch_sizes:
            for input_len in input_lengths:
                print(
                    f"\n=== Profiling: batch_size={batch_size}, input_length={input_len} ==="
                )

                prompt = _build_prompt_with_length(base_prompt, input_len)
                prompts = [prompt] * batch_size

                attn_pcts = []
                ffn_pcts = []
                other_pcts = []
                total_ms_list = []

                try:
                    for run_idx in range(num_runs):
                        print(
                            f"  -> 第 {run_idx + 1}/{num_runs} 轮 CUDA profile (bs={batch_size}, len={input_len})"
                        )
                        with torch.no_grad():
                            stats = _profile_once(llm, prompts, sampling_params)

                        if stats["total_us"] <= 0:
                            print("     本轮 profiler 未采集到有效 CUDA 时间，跳过。")
                            continue

                        attn_pct = 100.0 * stats["attn_us"] / stats["total_us"]
                        ffn_pct = 100.0 * stats["ffn_us"] / stats["total_us"]
                        other_pct = 100.0 * stats["other_us"] / stats["total_us"]
                        total_ms = stats["total_ms"]

                        attn_pcts.append(attn_pct)
                        ffn_pcts.append(ffn_pct)
                        other_pcts.append(other_pct)
                        total_ms_list.append(total_ms)

                        print(
                            f"     Attention {attn_pct:.2f}%  Linear {ffn_pct:.2f}%  Other {other_pct:.2f}%  total {total_ms:.2f} ms"
                        )

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or ("cuda" in msg and "oom" in msg):
                        print(
                            f"  [OOM] batch_size={batch_size}, input_length={input_len}，跳过该配置。"
                        )
                        torch.cuda.empty_cache()
                        continue
                    raise

                if not total_ms_list:
                    print(
                        f"  没有成功的 profile 轮次，跳过写入该配置 (bs={batch_size}, len={input_len})。"
                    )
                    continue

                avg_attn = float(np.mean(attn_pcts))
                avg_ffn = float(np.mean(ffn_pcts))
                avg_other = float(np.mean(other_pcts))
                avg_total_ms = float(np.mean(total_ms_list))

                print(
                    f"  平均: Attention {avg_attn:.2f}%  Linear {avg_ffn:.2f}%  Other {avg_other:.2f}%  total {avg_total_ms:.2f} ms"
                )

                f.write(
                    f"{batch_size},{input_len},{avg_attn:.6f},{avg_ffn:.6f},{avg_other:.6f},{avg_total_ms:.6f}\n"
                )
                f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-8B 生成阶段 CUDA 时间分解（Attention / FFN / Other）+ batch/length 扫描"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Qwen3-8B 模型路径（本地/HF地址）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请详细介绍大语言模型的注意力机制",
        help="基础提示词（用于拼接构造不同长度输入）",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100, help="生成的最大新 token 数"
    )
    parser.add_argument(
        "--num-runs", type=int, default=3, help="每个配置重复 profile 轮数（建议 3~5）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vllm_decode_sweep.csv",
        help="结果输出文件路径（CSV）",
    )
    args = parser.parse_args()

    sweep_decode_performance(
        model_path=args.model_path,
        base_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        output_path=args.output,
    )
