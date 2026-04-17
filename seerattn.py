
import argparse
import json
import time
from typing import Any

try:
    import torch
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: torch\n"
        "Install (example):\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "Or CPU-only:\n"
        "  pip install torch\n"
    ) from e

try:
    from transformers import AutoConfig, AutoTokenizer
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install:\n"
        "  pip install transformers\n"
    ) from e

try:
    from seer_attn import SeerAttnLlamaForCausalLM  # Sparse Prefill Modeling (optional)
    from seer_attn import SeerDecodingQwen3ForCausalLM  # Sparse Decoding Modeling
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: seer_attn (SeerAttention package)\n"
        "Install it according to the SeerAttention repo instructions, e.g.:\n"
        "  pip install -U seer-attn\n"
        "or\n"
        "  pip install -e .  (if you cloned the repo)\n"
    ) from e


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read_first_jsonl_record(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    raise ValueError(f"No valid JSONL record found in {path!r}")


def _iter_jsonl_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    xs = sorted(values)
    # Nearest-rank with linear interpolation.
    k = (len(xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


@torch.no_grad()
def benchmark_prefill_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # --- Prefill: one forward over the full prompt, producing KV cache.
    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # (B, 1)

    # --- Decode: incremental generation using cached KV.
    _cuda_sync_if_needed(device)
    t2 = time.perf_counter()
    cur_len = input_ids.shape[1]
    for _step in range(max_new_tokens):
        if attention_mask is not None:
            # Extend attention mask by one "1" for the new token.
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_len += 1
    _cuda_sync_if_needed(device)
    t3 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t3 - t2
    total_s = prefill_s + decode_s
    return {
        "prompt_tokens": float(input_ids.shape[1]),
        "new_tokens": float(max_new_tokens),
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "total_s": float(total_s),
        "decode_tok_per_s": float(max_new_tokens / decode_s) if decode_s > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates")
    parser.add_argument("--jsonl", default="media_long_context.jsonl")
    parser.add_argument("--field", default="content", help="JSON field to use as prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sparsity-method", default="token_budget", choices=["token_budget", "threshold"])
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", default="seerattn_bench_results.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit; otherwise test only first N records")
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="If >0 and using CUDA, call torch.cuda.empty_cache() every N records (can reduce OOM risk, may slow down)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # SeerAttention-R: sparse decoding
    config = AutoConfig.from_pretrained(args.model)
    base_model = getattr(config, "base_model", None) or args.model
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "seerattn_sparsity_method": args.sparsity_method,
    }
    if args.sparsity_method == "token_budget":
        model_kwargs["seerattn_token_budget"] = args.token_budget

    model = SeerDecodingQwen3ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)

    prefill_s_list: list[float] = []
    decode_s_list: list[float] = []
    total_s_list: list[float] = []
    decode_tps_list: list[float] = []
    prompt_tokens_list: list[float] = []

    n_ok = 0
    n_err = 0

    run_meta = {
        "model": args.model,
        "jsonl": args.jsonl,
        "field": args.field,
        "sparsity_method": args.sparsity_method,
        "token_budget": args.token_budget if args.sparsity_method == "token_budget" else None,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
    }

    started = time.perf_counter()
    with open(args.out, "w", encoding="utf-8") as out_f:
        for raw_idx, record in _iter_jsonl_records(args.jsonl):
            if args.limit and (n_ok + n_err) >= args.limit:
                break

            prompt = record.get(args.field)
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = json.dumps(record, ensure_ascii=False)

            try:
                stats = benchmark_prefill_decode(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )

                n_ok += 1
                prefill_s_list.append(float(stats["prefill_s"]))
                decode_s_list.append(float(stats["decode_s"]))
                total_s_list.append(float(stats["total_s"]))
                decode_tps_list.append(float(stats["decode_tok_per_s"]))
                prompt_tokens_list.append(float(stats["prompt_tokens"]))

                out_f.write(
                    json.dumps(
                        {
                            "type": "record",
                            "i": n_ok - 1,
                            "raw_idx": raw_idx,
                            **run_meta,
                            **stats,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception as e:
                n_err += 1
                out_f.write(
                    json.dumps(
                        {
                            "type": "error",
                            "i": n_ok + n_err - 1,
                            "raw_idx": raw_idx,
                            **run_meta,
                            "error": repr(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if args.empty_cache_every and device.type == "cuda":
                if (n_ok + n_err) % args.empty_cache_every == 0:
                    torch.cuda.empty_cache()

        elapsed = time.perf_counter() - started
        summary = {
            "type": "summary",
            **run_meta,
            "n_ok": n_ok,
            "n_err": n_err,
            "elapsed_s": float(elapsed),
            "prefill_s_mean": (sum(prefill_s_list) / len(prefill_s_list)) if prefill_s_list else None,
            "decode_s_mean": (sum(decode_s_list) / len(decode_s_list)) if decode_s_list else None,
            "total_s_mean": (sum(total_s_list) / len(total_s_list)) if total_s_list else None,
            "decode_tok_per_s_mean": (sum(decode_tps_list) / len(decode_tps_list)) if decode_tps_list else None,
            "prompt_tokens_mean": (sum(prompt_tokens_list) / len(prompt_tokens_list)) if prompt_tokens_list else None,
            "prefill_s_p50": _percentile(prefill_s_list, 50),
            "prefill_s_p90": _percentile(prefill_s_list, 90),
            "prefill_s_p99": _percentile(prefill_s_list, 99),
            "decode_s_p50": _percentile(decode_s_list, 50),
            "decode_s_p90": _percentile(decode_s_list, 90),
            "decode_s_p99": _percentile(decode_s_list, 99),
            "decode_tok_per_s_p50": _percentile(decode_tps_list, 50),
            "decode_tok_per_s_p10": _percentile(decode_tps_list, 10),
        }
        out_f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(json.dumps({"out": args.out, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


