import argparse
import json
from typing import Dict


def classify_kernel(name: str) -> str:
    """
    根据 CUDA kernel 名把时间粗分为 attn / ffn / other。
    这里按常见 FlashAttention / SDPA / GEMM 命名习惯写了一些规则，你可以根据实际 trace 再迭代。
    """
    n = (name or "").lower()

    # Attention 相关 kernel：FlashAttention / SDPA / QK^T / softmax 等
    attn_patterns = [
        "flashattn",
        "flash_attn",
        "flashattention",
        "scaled_dot_product_attention",
        "sdpa",
        "fwd_flash",
        "flash_fwd",
        "attn",
        "attention",
        "qk_matmul",
        "qk_softmax",
    ]

    # FFN 相关：MLP 里的 GEMM、激活
    ffn_patterns = [
        "mlp",
        "ffn",
        "feedforward",
        "feed_forward",
        "gate_proj",
        "up_proj",
        "down_proj",
        "silu",
        "swish",
        "gelu",
        "relu",
    ]

    if any(p in n for p in attn_patterns):
        return "attn"
    if any(p in n for p in ffn_patterns):
        return "ffn"

    # 一些 GEMM / matmul kernel 可能出现在 attn/ffn 内部，如果没有更精确信息，就先归 other
    return "other"


def parse_chrome_trace(path: str) -> Dict[str, float]:
    """
    解析从 torch.profiler 导出的 chrome trace json，
    按 kernel 名聚合时间，返回 {attn, ffn, other} 的 us 级时间和。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chrome trace 通常是一个包含 "traceEvents" 的 dict
    events = data.get("traceEvents", data)
    if not isinstance(events, list):
        raise ValueError("Unexpected chrome trace format: 'traceEvents' is not a list.")

    buckets = {"attn": 0.0, "ffn": 0.0, "other": 0.0}

    for ev in events:
        if not isinstance(ev, dict):
            continue

        # 只看 duration 型事件（'X'），并且 cat 里含有 'Kernel'（GPU kernel）
        ph = ev.get("ph")
        if ph != "X":
            continue

        cat = ev.get("cat", "")
        if "kernel" not in str(cat).lower():
            continue

        name = ev.get("name", "")
        dur = ev.get("dur", 0)  # us
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue

        cls = classify_kernel(name)
        buckets[cls] += float(dur)

    return buckets


def print_breakdown(buckets: Dict[str, float]) -> None:
    total_us = sum(buckets.values())
    if total_us <= 0:
        print("No GPU kernel time found in trace.")
        return

    print("\n===== vLLM Decode CUDA Kernel Breakdown (from trace.json) =====")
    for k in ["attn", "ffn", "other"]:
        us = buckets.get(k, 0.0)
        ms = us / 1000.0
        pct = us / total_us * 100.0
        print(f"{k:>5}: {ms:10.3f} ms  ({pct:6.2f}%)")
    print("total:", f"{total_us / 1000.0:10.3f} ms (kernel time sum)")
    print("===============================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parse vLLM torch.profiler chrome trace and compute attn/ffn/other breakdown."
    )
    parser.add_argument(
        "trace",
        type=str,
        help="Path to chrome trace json exported by torch.profiler (e.g. trace_vllm_decode.json).",
    )
    args = parser.parse_args()

    buckets = parse_chrome_trace(args.trace)
    print_breakdown(buckets)


if __name__ == "__main__":
    main()

