# # 默认：1000~20000 步长 2000，不保存 trace，只打印
# python profile_qwen3_decode.py

# # 自定义长度范围与步长
# python profile_qwen3_decode.py --length-min 2000 --length-max 10000 --length-step 2000

# # 需要保存 trace 时（对最后一个长度保存）
# python profile_qwen3_decode.py --trace decode_trace.json

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def build_messages(content: str) -> List[Dict[str, str]]:
    """构造 chat messages（可与原脚本一致或按需修改）。"""
    system = (
        "你将接收一段包含任务说明与输入信息的文本。"
        "请严格按照该文本中的规则生成结果，并且只输出最终的 ####总结内容。"
    )
    user = content.strip() + "\n\n请只输出####总结内容："
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _run_decode_steps_for_profiling(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_steps: int,
) -> None:
    """只跑 num_steps 步 decode，用于被 profiler 包裹。"""
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = prefill_outputs.past_key_values
    next_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(num_steps - 1):
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        with torch.no_grad():
            step_outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = step_outputs.past_key_values
        next_token = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _get_decoder_layers(model: torch.nn.Module):
    """获取 decoder 的 layers（Qwen/LLaMA 等通常为 model.model.layers）。"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return []


def profile_decode_phase_with_hooks(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_decode_steps: int = 20,
    silent: bool = False,
) -> Optional[Tuple[float, float, float, float]]:
    """用 forward hook 统计 decode 阶段各层 self_attn、mlp 及整段 decode 总耗时，Other = Total - Attention - FFN。
    若 silent=True，只打印一行并返回 (attn_pct, ffn_pct, other_pct, total_ms)；否则打印完整表格并返回 None。
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    layers = _get_decoder_layers(model)
    if not layers:
        print("  [Hook] 未找到 model.model.layers，跳过 hook 计时。")
        return None

    attn_times: List[float] = [0.0] * len(layers)
    ffn_times: List[float] = [0.0] * len(layers)
    handles = []

    def make_hook(layer_idx: int, t_list: List[float]):
        t_start = [0.0]

        def pre_hook(module, inp):
            torch.cuda.synchronize()
            t_start[0] = time.perf_counter()

        def post_hook(module, inp, out):
            torch.cuda.synchronize()
            t_list[layer_idx] += time.perf_counter() - t_start[0]

        return pre_hook, post_hook

    # 先做 prefill（不挂 hook），只为了得到 past_key_values 和第一个 token
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = prefill_outputs.past_key_values
    next_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # 仅对 decode 阶段挂 hook，这样 Attention/FFN 占比才是「decode 总时间」中的占比
    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            pre, post = make_hook(i, attn_times)
            handles.append(layer.self_attn.register_forward_pre_hook(pre))
            handles.append(layer.self_attn.register_forward_hook(post))
        if hasattr(layer, "mlp"):
            pre, post = make_hook(i, ffn_times)
            handles.append(layer.mlp.register_forward_pre_hook(pre))
            handles.append(layer.mlp.register_forward_hook(post))

    torch.cuda.synchronize()
    t_decode_start = time.perf_counter()
    for _ in range(num_decode_steps - 1):
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        with torch.no_grad():
            step_outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = step_outputs.past_key_values
        next_token = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    total_decode_sec = time.perf_counter() - t_decode_start

    for h in handles:
        h.remove()

    total_attn = sum(attn_times)
    total_ffn = sum(ffn_times)
    total_decode_ms = total_decode_sec * 1000
    other_sec = max(0.0, total_decode_sec - total_attn - total_ffn)
    other_ms = other_sec * 1000
    attn_ms = total_attn * 1000
    ffn_ms = total_ffn * 1000

    if total_decode_sec <= 0:
        return None
    attn_pct = 100.0 * total_attn / total_decode_sec
    ffn_pct = 100.0 * total_ffn / total_decode_sec
    other_pct = 100.0 * other_sec / total_decode_sec

    num_actual_decode_steps = num_decode_steps - 1  # prefill 已产生 1 个 token，后续只跑 num_decode_steps-1 次 decode
    if silent:
        print(f"    prompt_len={input_ids.shape[1]:>5}  Attention {attn_pct:5.1f}%  FFN {ffn_pct:5.1f}%  Other {other_pct:5.1f}%  total {total_decode_ms:.1f} ms")
        return (attn_pct, ffn_pct, other_pct, total_decode_ms)

    print("\n" + "=" * 60)
    print("Decode 阶段耗时 (Forward Hook: 整段 decode = Attention + FFN + Other)")
    print(f"  共 {num_actual_decode_steps} 次 decode 步（不含 prefill）")
    print("=" * 60)
    print(f"  {'类别':<12}  {'时间(ms)':>12}  {'占比':>8}")
    print("-" * 60)
    print(f"  {'Attention':<12}  {attn_ms:>12.2f}  {attn_pct:>7.1f}%")
    print(f"  {'FFN':<12}  {ffn_ms:>12.2f}  {ffn_pct:>7.1f}%")
    print(f"  {'Other':<12}  {other_ms:>12.2f}  {other_pct:>7.1f}%")
    print("-" * 60)
    print(f"  {'Total':<12}  {total_decode_ms:>12.2f}  100.0%")
    print(f"  单步 decode 均时: {total_decode_ms / num_actual_decode_steps:.2f} ms")
    print("=" * 60)
    return None


def _classify_by_stack_or_key(stack: List[str], key: str) -> str:
    """根据调用栈或 key 将 CUDA 事件归为 attention / ffn / other（Qwen3 通用）。"""
    full = " ".join(stack).lower() if stack else ""
    k = (full or (key or "")).lower()
    if "self_attn" in k:
        return "attention"
    if "mlp" in k or "gate_proj" in k or "up_proj" in k or "down_proj" in k:
        return "ffn"
    if not full and key:
        if "flash" in k and "fwd" in k:
            return "attention"
        if "silu" in k or "swish" in k:
            return "ffn"
    return "other"


def profile_decode_phase(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_decode_steps: int = 20,
    trace_path: Optional[str] = "decode_trace.json",
    group_by_stack_n: int = 12,
) -> None:
    """对 decode 阶段做 CUDA profiling，输出 Attention/FFN/Other 占比并导出 Chrome trace。"""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
    ) as prof:
        _run_decode_steps_for_profiling(
            model, input_ids, attention_mask, num_decode_steps
        )

    torch.cuda.synchronize()

    cuda_attention_us = 0.0
    cuda_ffn_us = 0.0
    cuda_other_us = 0.0
    by_category: Dict[str, List[Tuple[str, float]]] = {
        "attention": [],
        "ffn": [],
        "other": [],
    }

    def _get_cuda_time(evt) -> float:
        return (
            getattr(evt, "device_time_total", None)
            or getattr(evt, "cuda_time_total", None)
            or getattr(evt, "self_device_time_total", None)
            or 0
        )

    try:
        raw_events = prof.events()
    except Exception:
        raw_events = []

    if raw_events:
        for evt in raw_events:
            cuda_time = _get_cuda_time(evt)
            if cuda_time == 0:
                continue
            stack = getattr(evt, "stack", None) or []
            if isinstance(stack, str):
                stack = [stack]
            key = str(getattr(evt, "key", "") or "")
            cat = _classify_by_stack_or_key(stack, key)
            label = key[:100] if key else (stack[0][:100] if stack else "")
            if cat == "attention":
                cuda_attention_us += cuda_time
                by_category["attention"].append((label, cuda_time))
            elif cat == "ffn":
                cuda_ffn_us += cuda_time
                by_category["ffn"].append((label, cuda_time))
            else:
                cuda_other_us += cuda_time
                by_category["other"].append((label, cuda_time))
    else:
        try:
            key_averages = prof.key_averages(group_by_stack_n=group_by_stack_n)
        except Exception:
            key_averages = prof.key_averages()
        for evt in key_averages:
            cuda_time = _get_cuda_time(evt)
            if cuda_time == 0:
                continue
            key = str(evt.key or "")
            stack = getattr(evt, "stack", None) or []
            if isinstance(stack, str):
                stack = [stack]
            cat = _classify_by_stack_or_key(stack, key)
            if cat == "attention":
                cuda_attention_us += cuda_time
                by_category["attention"].append((key[:120], cuda_time))
            elif cat == "ffn":
                cuda_ffn_us += cuda_time
                by_category["ffn"].append((key[:120], cuda_time))
            else:
                cuda_other_us += cuda_time
                by_category["other"].append((key[:120], cuda_time))

    total_us = cuda_attention_us + cuda_ffn_us + cuda_other_us
    total_ms = total_us / 1000.0

    print("\n" + "=" * 60)
    print("Decode 阶段 CUDA Kernel 时间汇总 (按调用栈归类)")
    print(f"  共 profile {num_decode_steps} 步 decode")
    print("=" * 60)
    print(f"  {'类别':<12}  {'时间(ms)':>12}  {'占比':>8}")
    print("-" * 60)
    if total_ms > 0:
        for label, us in [
            ("Attention", cuda_attention_us),
            ("FFN", cuda_ffn_us),
            ("Other", cuda_other_us),
        ]:
            pct = 100.0 * us / total_us
            print(f"  {label:<12}  {us/1000:>12.2f}  {pct:>7.1f}%")
        print("-" * 60)
        print(f"  {'Total':<12}  {total_ms:>12.2f}  100.0%")
        print(f"  单步 decode 均时: {total_ms / num_decode_steps:.2f} ms")
    print("=" * 60)

    for cat in ("attention", "ffn"):
        entries = by_category[cat]
        if not entries:
            continue
        entries.sort(key=lambda x: -x[1])
        print(f"\n--- Top {min(5, len(entries))} {cat.upper()} kernels (by CUDA time) ---")
        for key_snippet, t_us in entries[:5]:
            print(f"  {t_us/1000:.2f} ms  {key_snippet}")

    other_entries = by_category["other"]
    if other_entries:
        other_entries.sort(key=lambda x: -x[1])
        print("\n--- Other 包含 (未归入 Attention/FFN 的 CUDA 时间) ---")
        print("  典型: GEMM/矩阵乘、LayerNorm、拷贝/索引、Reduce 等。")
        print(f"  Top {min(8, len(other_entries))} Other kernels (by CUDA time):")
        for key_snippet, t_us in other_entries[:8]:
            print(f"    {t_us/1000:.2f} ms  {key_snippet}")

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print("\n" + "=" * 60)
        print("如何可视化 trace JSON")
        print("=" * 60)
        print("  1. 用 Chrome 打开: chrome://tracing")
        print("  2. 点击左上角 「Load」 或拖拽 JSON 文件到页面")
        print(f"  3. 选择本机路径: {trace_path}")
        print("  即可看到时间轴上的 CUDA kernel、CPU 调用等，可缩放、点选查看详情。")
        print("=" * 60)
    print("\n  (上方的 Forward Hook 结果为权威占比；CUDA Kernel 汇总多为 Other，仅作 kernel 级参考)")


def _truncate_or_pad_to_length(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    length: int,
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 input_ids / attention_mask 截断或左填充到指定长度。"""
    cur_len = input_ids.shape[1]
    if cur_len >= length:
        return input_ids[:, :length].contiguous(), (attention_mask[:, :length] if attention_mask is not None else torch.ones(1, length, dtype=torch.long, device=device))
    # 左填充到 length
    pad_len = length - cur_len
    new_ids = torch.full((input_ids.shape[0], length), pad_token_id, dtype=input_ids.dtype, device=device)
    new_ids[:, pad_len:] = input_ids
    new_mask = torch.zeros(input_ids.shape[0], length, dtype=torch.long, device=device)
    if attention_mask is not None:
        new_mask[:, pad_len:] = attention_mask
    else:
        new_mask[:, pad_len:] = 1
    return new_ids, new_mask


def main(
    jsonl_path: str = "media_long_context.jsonl",
    model_name: str = "Qwen/Qwen3-8B",
    profile_steps: int = 20,
    warmup_steps: int = 5,
    length_min: int = 1000,
    length_max: int = 20000,
    length_step: int = 2000,
    trace_path: Optional[str] = None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).cuda()
    device = next(model.parameters()).device

    with open(jsonl_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"文件 {jsonl_path} 为空或首行为空")

    example = json.loads(line)
    prompt = example.get("content", "")
    messages = build_messages(prompt)
    if not messages:
        raise ValueError("样本中缺少 content 字段或为空")

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids_full = inputs["input_ids"].to(device)
    attention_mask_full = inputs.get("attention_mask")
    if attention_mask_full is not None:
        attention_mask_full = attention_mask_full.to(device)

    lengths = list(range(length_min, length_max + 1, length_step))
    batch_sizes = [1, 2, 3, 4]
    print(f"Profile 不同 prompt 长度 × batch size 下的 decode 占比 (lengths: {length_min} ~ {length_max}, step {length_step}, batch_sizes: {batch_sizes})")
    print(f"  长度序列: {lengths}")
    print(f"  profile_steps={profile_steps}, warmup_steps={warmup_steps}")
    if trace_path:
        print(f"  将保存 Chrome trace: {trace_path}")
    else:
        print("  不保存 trace，仅打印结果。")
    print()

    # (length, batch_size, attn_pct, ffn_pct, other_pct, total_ms)
    results: List[Tuple[int, int, float, float, float, float]] = []

    for batch_size in batch_sizes:
        for i, length in enumerate(lengths):
            input_ids, attention_mask = _truncate_or_pad_to_length(
                input_ids_full, attention_mask_full, length, pad_token_id, device
            )
            # 扩展到当前 batch_size（单样本重复）
            if batch_size > 1:
                input_ids = input_ids.repeat(batch_size, 1)
                attention_mask = attention_mask.repeat(batch_size, 1)
            print(f"  [bs={batch_size}] [{i+1}/{len(lengths)}] prompt_len={length}")

            try:
                with torch.no_grad():
                    if warmup_steps > 0:
                        _run_decode_steps_for_profiling(model, input_ids, attention_mask, warmup_steps)
                        torch.cuda.synchronize()

                    row = profile_decode_phase_with_hooks(
                        model,
                        input_ids,
                        attention_mask,
                        num_decode_steps=profile_steps,
                        silent=True,
                    )
                if row is not None:
                    attn_pct, ffn_pct, other_pct, total_ms = row
                    results.append((length, batch_size, attn_pct, ffn_pct, other_pct, total_ms))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    print(f"    -> OOM 跳过 (prompt_len={length}, batch_size={batch_size})")
                else:
                    raise

    # 汇总表（整段 decode = Attention + FFN + Other）
    print()
    print("=" * 95)
    print("汇总: 不同 prompt 长度 × batch size 下的 decode 计算占比 (Attention + FFN + Other = 100%)")
    print("=" * 95)
    print(f"  {'prompt_len':>10}  {'batch_size':>10}  {'Attention%':>10}  {'FFN%':>10}  {'Other%':>10}  {'total(ms)':>12}")
    print("-" * 95)
    for length, batch_size, attn_pct, ffn_pct, other_pct, total_ms in results:
        print(f"  {length:>10}  {batch_size:>10}  {attn_pct:>10.1f}  {ffn_pct:>10.1f}  {other_pct:>10.1f}  {total_ms:>12.1f}")
    print("=" * 95)

    # 按 batch_size 分组，绘制 decode 时 Attention 占比随 prompt_len 变化的折线图
    fig, ax = plt.subplots(figsize=(10, 6))
    for bs in batch_sizes:
        points = [(r[0], r[2]) for r in results if r[1] == bs]  # (prompt_len, attn_pct)
        points.sort(key=lambda x: x[0])
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=f"batch_size={bs}", linewidth=2, markersize=6)
    ax.set_xlabel("Prompt length (tokens)", fontsize=12)
    ax.set_ylabel("Decode Attention 占比 (%)", fontsize=12)
    ax.set_title("Decode 阶段 Attention 计算占比 vs Prompt 长度 (不同 batch size)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    out_plot = "profile_qwen3_decode_attn_pct_vs_prompt_len.png"
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n折线图已保存: {out_plot}")

    # 保存所有测试结果到 JSON
    out_json = "profile_qwen3_decode_results.json"
    save_data = [
        {
            "prompt_len": length,
            "batch_size": batch_size,
            "attention_pct": attn_pct,
            "ffn_pct": ffn_pct,
            "other_pct": other_pct,
            "total_ms": total_ms,
        }
        for length, batch_size, attn_pct, ffn_pct, other_pct, total_ms in results
    ]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"测试结果已保存: {out_json}")

    # 若指定了 trace，对最后一个长度跑一次 CUDA profiler 并保存
    if trace_path and results:
        length_last = results[-1][0]
        input_ids, attention_mask = _truncate_or_pad_to_length(
            input_ids_full, attention_mask_full, length_last, pad_token_id, device
        )
        try:
            with torch.no_grad():
                if warmup_steps > 0:
                    _run_decode_steps_for_profiling(model, input_ids, attention_mask, warmup_steps)
                    torch.cuda.synchronize()
                print(f"\n保存 Chrome trace (prompt_len={length_last}): {trace_path}")
                profile_decode_phase(
                    model,
                    input_ids,
                    attention_mask,
                    num_decode_steps=profile_steps,
                    trace_path=trace_path,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                print(f"\n保存 trace 时 OOM，跳过 (prompt_len={length_last})")
            else:
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Qwen3-8B decode 阶段 (Attention vs FFN)")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="media_long_context.jsonl",
        help="输入 jsonl 路径，取首行 content 作为 prompt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="模型名 (默认 Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="profile 时运行的 decode 步数",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Chrome trace 输出路径；默认不保存，仅打印",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="每个长度 profile 前的 warmup decode 步数 (默认 5)，设为 0 可关闭",
    )
    parser.add_argument(
        "--length-min",
        type=int,
        default=1000,
        help="profile 的 prompt 长度下界 (token 数)",
    )
    parser.add_argument(
        "--length-max",
        type=int,
        default=20000,
        help="profile 的 prompt 长度上界 (token 数)",
    )
    parser.add_argument(
        "--length-step",
        type=int,
        default=2000,
        help="prompt 长度步长 (默认 2000，即 1000,3000,...,19000)",
    )
    args = parser.parse_args()
    main(
        jsonl_path=args.jsonl,
        model_name=args.model,
        profile_steps=args.profile_steps,
        warmup_steps=args.warmup_steps,
        length_min=args.length_min,
        length_max=args.length_max,
        length_step=args.length_step,
        trace_path=args.trace,
    )
