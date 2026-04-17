#  python /home/wangx/haixin_long_context/seerattn_batch_bench.py \
#   --impl seer_decode \
#   --model SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates \
#   --max-new-tokens 512 --batch-sizes 1,2,4,8,16 \
#   --out /home/wangx/haixin_long_context/seer_decode_batch_bench.json 

#  python /home/wangx/haixin_long_context/seerattn_batch_bench.py \
#   --impl hf_causal_lm \
#   --model Qwen/Qwen3-8B \
#   --max-new-tokens 512 --batch-sizes 1,2,4,8,16 \
#   --out /home/wangx/haixin_long_context/qwen3_batch_bench.json

import argparse
import json
import os
import time
from typing import Any

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


@torch.no_grad()
def bench_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    texts = [prompt] * batch_size
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    prompt_tokens = float(input_ids.shape[1])
    total_prompt_tokens = float(input_ids.numel())

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
    _cuda_sync_if_needed(device)
    t2 = time.perf_counter()
    for _ in range(max_new_tokens):
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
    _cuda_sync_if_needed(device)
    t3 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t3 - t2

    total_new_tokens = float(batch_size * max_new_tokens)
    return {
        "batch_size": float(batch_size),
        "prompt_tokens_per_sample": prompt_tokens,
        "prompt_tokens_total": total_prompt_tokens,
        "new_tokens_per_sample": float(max_new_tokens),
        "new_tokens_total": total_new_tokens,
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "total_s": float(prefill_s + decode_s),
        "prefill_prompt_tok_per_s": float(total_prompt_tokens / prefill_s) if prefill_s > 0 else float("inf"),
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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sparsity-method", default="token_budget", choices=["token_budget", "threshold"])
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--out", default="seerattn_batch_bench.json")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
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
        if args.sparsity_method == "token_budget":
            model_kwargs["seerattn_token_budget"] = args.token_budget
        model = SeerDecodingQwen3ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    else:
        # Vanilla HF baseline, e.g. Qwen/Qwen3-8B
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(device)

    record = _read_first_jsonl_record(args.jsonl)
    prompt = record.get(args.field)
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = json.dumps(record, ensure_ascii=False)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    results: list[dict[str, Any]] = []

    run_meta = {
        "model": args.model,
        "impl": args.impl,
        "tokenizer": tokenizer_src,
        "jsonl": args.jsonl,
        "field": args.field,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "sparsity_method": args.sparsity_method if args.impl == "seer_decode" else None,
        "token_budget": (args.token_budget if args.sparsity_method == "token_budget" else None) if args.impl == "seer_decode" else None,
        "batch_sizes": batch_sizes,
        "note": args.note,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    for bs in batch_sizes:
        try:
            stats = bench_batch(
                model,
                tokenizer,
                prompt,
                batch_size=bs,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            results.append({"type": "ok", **stats})
        except torch.OutOfMemoryError as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results.append({"type": "oom", "batch_size": bs, "error": repr(e)})
        except Exception as e:
            results.append({"type": "error", "batch_size": bs, "error": repr(e)})

    payload = {"meta": run_meta, "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

