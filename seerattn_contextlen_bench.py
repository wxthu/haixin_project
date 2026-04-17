# python seerattn_contextlen_bench.py \
#   --impl seer_decode \
#   --model SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates \
#   --jsonl media_long_context.jsonl --field content \
#   --max-new-tokens 512 \
#   --repeats 3 \
#   --out seer_decode_ctxlen_bench.json
  
# python seerattn_contextlen_bench.py \
#   --impl hf_causal_lm \
#   --model Qwen/Qwen3-8B \
#   --jsonl media_long_context.jsonl --field content \
#   --max-new-tokens 512 \
#   --repeats 3 \
#   --out hf_ctxlen_bench.json
  
import argparse
import json
import os
import time
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: torch\n"
        "Install torch in your environment, then retry.\n"
    ) from e

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install:\n"
        "  pip install transformers\n"
    ) from e

try:
    from seer_attn import SeerDecodingQwen3ForCausalLM
except ModuleNotFoundError:
    SeerDecodingQwen3ForCausalLM = None  # type: ignore[assignment]


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


def _iter_context_lengths(start: int, end: int, step: int) -> Iterable[int]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")
    x = start
    while x <= end:
        yield x
        x += step


def _write_payload(path: str, meta: dict[str, Any], results: list[dict[str, Any]]) -> None:
    payload = {"meta": meta, "results": results}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _parse_int_list(spec: str) -> list[int]:
    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("Empty list spec")
    return items


def _build_input_ids_for_context_len(
    tokenizer: Any,
    base_text: str,
    *,
    target_ctx_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if target_ctx_len <= 0:
        raise ValueError("target_ctx_len must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    base_ids_list: list[int] = tokenizer.encode(base_text, add_special_tokens=False)
    if len(base_ids_list) == 0:
        # Fallback to a non-empty token sequence (e.g. EOS)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("Tokenizer produced empty ids and has no eos_token_id.")
        base_ids_list = [int(eos)]

    base_len = len(base_ids_list)
    if base_len >= target_ctx_len:
        ids_list = base_ids_list[:target_ctx_len]
        mode = "truncate"
        repeats = 1
    else:
        repeats = (target_ctx_len + base_len - 1) // base_len
        ids_list = (base_ids_list * repeats)[:target_ctx_len]
        mode = "repeat"

    input_ids_1 = torch.tensor([ids_list], dtype=torch.long, device=device)  # (1, T)
    attention_mask_1 = torch.ones_like(input_ids_1, dtype=torch.long, device=device)  # (1, T)
    if batch_size == 1:
        input_ids = input_ids_1
        attention_mask = attention_mask_1
    else:
        input_ids = input_ids_1.repeat(batch_size, 1)  # (B, T)
        attention_mask = attention_mask_1.repeat(batch_size, 1)  # (B, T)
    meta = {
        "base_tokens": int(base_len),
        "target_ctx_len": int(target_ctx_len),
        "mode": mode,
        "repeats": int(repeats),
        "batch_size": int(batch_size),
    }
    return input_ids, attention_mask, meta


@torch.no_grad()
def bench_context_len(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    prompt_tokens_total = float(input_ids.numel())
    prompt_len = int(input_ids.shape[1])
    batch_size = int(input_ids.shape[0])

    # --- Prefill
    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # (B, 1)

    # --- Decode
    # Avoid per-step torch.cat on attention_mask (O(T^2) allocations).
    # Pre-allocate a full mask and slice per step.
    attn_full = torch.ones(
        (attention_mask.shape[0], prompt_len + max_new_tokens),
        device=device,
        dtype=attention_mask.dtype,
    )
    attn_full[:, :prompt_len] = attention_mask
    _cuda_sync_if_needed(device)
    t2 = time.perf_counter()
    for i in range(max_new_tokens):
        step_attn = attn_full[:, : (prompt_len + i + 1)]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(
                input_ids=next_token,
                attention_mask=step_attn,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    _cuda_sync_if_needed(device)
    t3 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t3 - t2
    total_new_tokens = float(batch_size * max_new_tokens)
    return {
        "prompt_tokens_total": float(prompt_tokens_total),
        "new_tokens_total": float(total_new_tokens),
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "total_s": float(prefill_s + decode_s),
        "prefill_prompt_tok_per_s": float(prompt_tokens_total / prefill_s) if prefill_s > 0 else float("inf"),
        "decode_new_tok_per_s": float(total_new_tokens / decode_s) if decode_s > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates")
    parser.add_argument(
        "--impl",
        default="seer_decode",
        choices=["seer_decode", "hf_causal_lm"],
        help="Model implementation to benchmark: SeerAttention sparse decode vs vanilla HF causal LM",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer repo/path. If empty, uses config.base_model (if present) else --model",
    )
    parser.add_argument("--jsonl", default="media_long_context.jsonl")
    parser.add_argument("--field", default="content")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--batch-sizes",
        default="1,2,3,4",
        help="Batch sizes to benchmark (comma-separated), e.g. 1,2,4",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sparsity-method", default="token_budget", choices=["token_budget", "threshold"])
    parser.add_argument(
        "--context-len-start",
        type=int,
        default=2000,
        help="Start context length (tokens)",
    )
    parser.add_argument(
        "--context-len-end",
        type=int,
        default=20000,
        help="End context length (tokens, inclusive)",
    )
    parser.add_argument(
        "--context-len-step",
        type=int,
        default=2000,
        help="Context length step (tokens)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations per context length (not recorded)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured repeats per context length (averaged)",
    )
    parser.add_argument("--out", default="seerattn_contextlen_bench.json")
    parser.add_argument(
        "--note",
        default="",
        help="Optional note string saved into output (e.g. gpu model / run tag)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    config = AutoConfig.from_pretrained(args.model)
    default_tokenizer_src = getattr(config, "base_model", None) or args.model
    tokenizer_src = args.tokenizer.strip() or default_tokenizer_src
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.impl == "seer_decode":
        if SeerDecodingQwen3ForCausalLM is None:
            raise SystemExit(
                "seer_attn is not installed but --impl seer_decode was requested.\n"
                "Install SeerAttention (seer_attn) then retry, or use --impl hf_causal_lm.\n"
            )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "seerattn_sparsity_method": args.sparsity_method,
        }
        model = SeerDecodingQwen3ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(device)

    record = _read_first_jsonl_record(args.jsonl)
    prompt = record.get(args.field)
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = json.dumps(record, ensure_ascii=False)

    ctx_lens = list(_iter_context_lengths(args.context_len_start, args.context_len_end, args.context_len_step))
    batch_sizes = _parse_int_list(args.batch_sizes)
    results: list[dict[str, Any]] = []

    run_meta: dict[str, Any] = {
        "model": args.model,
        "impl": args.impl,
        "tokenizer": tokenizer_src,
        "jsonl": args.jsonl,
        "field": args.field,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "sparsity_method": args.sparsity_method if args.impl == "seer_decode" else None,
        "context_lens": ctx_lens,
        "batch_sizes": batch_sizes,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "note": args.note,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    for ctx in ctx_lens:
        # For sparse decode with token_budget, keep it = context_len/2 (per user request).
        if args.impl == "seer_decode" and args.sparsity_method == "token_budget":
            setattr(model.config, "seerattn_token_budget", int(ctx // 2))

        for bs in batch_sizes:
            input_ids, attention_mask, build_meta = _build_input_ids_for_context_len(
                tokenizer, prompt, target_ctx_len=ctx, batch_size=bs, device=device
            )

            # Warmup
            for _ in range(max(0, int(args.warmup))):
                _ = bench_context_len(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )

            # Measured repeats
            prefill_s_sum = 0.0
            decode_s_sum = 0.0
            prefill_tokps_sum = 0.0
            decode_tokps_sum = 0.0
            ok = True
            err: str | None = None
            nrep = max(1, int(args.repeats))
            try:
                for _ in range(nrep):
                    stats = bench_context_len(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
                    prefill_s_sum += float(stats["prefill_s"])
                    decode_s_sum += float(stats["decode_s"])
                    prefill_tokps_sum += float(stats["prefill_prompt_tok_per_s"])
                    decode_tokps_sum += float(stats["decode_new_tok_per_s"])
            except torch.OutOfMemoryError as e:
                ok = False
                err = repr(e)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                ok = False
                err = repr(e)

            if ok:
                results.append(
                    {
                        "type": "ok",
                        "context_len": int(ctx),
                        "batch_size": int(bs),
                        "token_budget": (
                            int(ctx // 2)
                            if args.impl == "seer_decode" and args.sparsity_method == "token_budget"
                            else None
                        ),
                        **build_meta,
                        "prompt_tokens_total": float(input_ids.numel()),
                        "new_tokens_total": float(bs * args.max_new_tokens),
                        "prefill_s_avg": float(prefill_s_sum / nrep),
                        "decode_s_avg": float(decode_s_sum / nrep),
                        "prefill_prompt_tok_per_s_avg": float(prefill_tokps_sum / nrep),
                        "decode_new_tok_per_s_avg": float(decode_tokps_sum / nrep),
                    }
                )
            else:
                results.append(
                    {
                        "type": "oom" if err and "OutOfMemoryError" in err else "error",
                        "context_len": int(ctx),
                        "batch_size": int(bs),
                        "token_budget": (
                            int(ctx // 2)
                            if args.impl == "seer_decode" and args.sparsity_method == "token_budget"
                            else None
                        ),
                        **build_meta,
                        "error": err,
                    }
                )

            # Incrementally persist results so that if the process is killed (e.g. OOM),
            # all completed (context_len, batch_size) runs are still saved.
            _write_payload(args.out, run_meta, results)

    # Final print (stdout) with whatever we have.
    print(json.dumps({"meta": run_meta, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

