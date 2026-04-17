import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from llmlingua import PromptCompressor
from openai import OpenAI
from tqdm import tqdm

# LongBench v1 prompt templates (THUDM/LongBench config/dataset2prompt.json).
LONGBENCH_V1_DATASET_PROMPTS: Dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结："
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "lsht": ("请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}"),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are after "
        "removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph "
        "the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be "
        "like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n"
        "{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是："
    ),
    "lcc": ("Please complete the code given below. \n{context}Next line of code:\n"),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
}


def _prompt_task_key(task: str) -> Optional[str]:
    """Map local JSONL task name (e.g. hotpotqa_e) to LongBench v1 prompt key."""
    if task in LONGBENCH_V1_DATASET_PROMPTS:
        return task
    if task.endswith("_e"):
        base = task[:-2]
        if base in LONGBENCH_V1_DATASET_PROMPTS:
            return base
    return None


def _format_longbench_prompt(template: str, context: str, input_text: str) -> str:
    """Fill {context} and {input} without str.format (context may contain braces)."""
    return template.replace("{context}", context).replace("{input}", input_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LongBench v1 local JSONL tasks via OpenAI-compatible vLLM, "
            "with structured prompt compression (compress context, keep question intact)."
        )
    )
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--base-url", type=str, default="http://localhost:8090/v1")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="longbench_v1_datasets",
        help="Directory containing LongBench v1 *.jsonl files.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task names without .jsonl suffix, or 'all'.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="If > 0, run only first N selected tasks.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=0,
        help="If > 0, run only first N samples per task.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass extra_body={'enable_thinking': True} for Qwen3-like models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="longbench_v1_results_elbow",
        help="Output directory; one jsonl result file per task.",
    )

    # Compression controls (reuse vllm_serve_compressed_context_openai.py strategy).
    parser.add_argument(
        "--compressor-model",
        type=str,
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        help="LLMLingua compressor model name.",
    )
    parser.add_argument(
        "--compression-rate",
        type=float,
        default=0.5,
        help="Target compression rate for context text.",
    )
    parser.add_argument(
        "--min-context-chars",
        type=int,
        default=600,
        help="Skip compression when context is shorter than this threshold.",
    )
    parser.add_argument(
        "--disable-compression",
        action="store_true",
        help="Run same pipeline but do not compress context.",
    )
    return parser.parse_args()


def list_task_files(data_dir: str) -> Dict[str, str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    task2path: Dict[str, str] = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".jsonl"):
            continue
        task = name[:-6]
        task2path[task] = os.path.join(data_dir, name)
    if not task2path:
        raise RuntimeError(f"No .jsonl files found in {data_dir}")
    return task2path


def resolve_tasks(args: argparse.Namespace, task2path: Dict[str, str]) -> List[str]:
    if args.tasks.strip().lower() == "all":
        selected = sorted(task2path.keys())
    else:
        selected = [x.strip() for x in args.tasks.split(",") if x.strip()]
        missing = [t for t in selected if t not in task2path]
        if missing:
            raise ValueError(
                f"Tasks not found in {args.data_dir}: {missing}. "
                f"Available tasks: {sorted(task2path.keys())}"
            )
    if args.max_tasks > 0:
        selected = selected[: args.max_tasks]
    return selected


def load_jsonl(path: str, max_samples: int = 0) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples > 0 and len(records) >= max_samples:
                break
    return records


def split_context_blocks(context: str) -> List[str]:
    """
    Structured split for context compression.
    Keep paragraph-like structure and list markers to avoid over-fragmentation.
    """
    text = context.strip()
    if not text:
        return []

    # Prefer splitting by blank lines first.
    blocks = [x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()]
    if len(blocks) > 1:
        return blocks

    # If no blank-line structure, use soft split by numbered/heading markers.
    markers = r"(?=(?:\n(?:Paragraph|段落|Doc|Document)\s*\d+[:：]|^#+\s|^\d+[.)]\s))"
    parts = [x.strip() for x in re.split(markers, text, flags=re.M) if x.strip()]
    return parts if parts else [text]


def compress_context_structured(
    compressor: PromptCompressor,
    context: str,
    question: str,
    rate: float,
) -> Tuple[str, Dict[str, Optional[float]]]:
    """
    Compress only context; keep question untouched.
    Tries block-level compression first, falls back to whole context.
    """
    if not context.strip():
        return context, {
            "compression_rate": None,
            "original_context_chars": 0,
            "compressed_context_chars": 0,
            "compression_time_seconds": 0.0,
            "compression_error": None,
        }

    start = time.perf_counter()
    original_len = len(context)
    block_rates: List[float] = []
    compression_error: Optional[str] = None

    try:
        blocks = split_context_blocks(context)
        compressed_blocks: List[str] = []
        for blk in blocks:
            if not blk.strip():
                continue
            comp = compressor.compress_prompt(
                blk,
                question=question if question else "question",
                rate=rate,
                adaptive_compression=True,
                adaptive_strategy="elbow",
                adaptive_mass_alpha=0.75,
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.0,
                condition_compare=True,
                context_budget="+100",
                rank_method="longllmlingua",
            )
            compressed_blocks.append(comp["compressed_prompt"])
            if "compression_rate" in comp:
                block_rates.append(float(comp["compression_rate"]))

        compressed_context = "\n\n".join(compressed_blocks).strip()
        if not compressed_context:
            compressed_context = context
    except Exception as exc:  # Keep evaluation running even if one sample fails compression.
        compression_error = str(exc)
        try:
            comp = compressor.compress_prompt(
                context,
                question=question if question else "question",
                rate=rate,
                adaptive_compression=True,
                adaptive_strategy="elbow",
                adaptive_mass_alpha=0.75,
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.0,
                condition_compare=True,
                context_budget="+100",
                rank_method="longllmlingua",
            )
            compressed_context = comp["compressed_prompt"]
            if "compression_rate" in comp:
                block_rates.append(float(comp["compression_rate"]))
        except Exception as exc2:
            compression_error = f"{compression_error} | fallback_failed={exc2}"
            compressed_context = context

    compressed_len = len(compressed_context)
    elapsed = time.perf_counter() - start
    rate_agg: Optional[float]
    if block_rates:
        rate_agg = sum(block_rates) / len(block_rates)
    else:
        rate_agg = (
            (compressed_len / original_len) if original_len > 0 else None
        )

    return compressed_context, {
        "compression_rate": round(rate_agg, 6) if rate_agg is not None else None,
        "original_context_chars": original_len,
        "compressed_context_chars": compressed_len,
        "compression_time_seconds": round(elapsed, 4),
        "compression_error": compression_error,
    }


def build_messages_generic(sample: Dict) -> List[Dict[str, str]]:
    """Fallback when no LongBench v1 template exists for this task name."""
    context = str(sample.get("context", "")).strip()
    user_input = str(sample.get("input", "")).strip()
    all_classes = sample.get("all_classes", None)
    language = str(sample.get("language", "")).lower()

    if language == "zh" or "_zh" in str(sample.get("dataset", "")):
        system_prompt = (
            "你是一个长上下文任务助手。"
            "请严格基于给定上下文回答。"
            "如果是分类任务，答案必须从候选类别中选择。"
            "只输出最终答案，不要额外解释。"
        )
    else:
        system_prompt = (
            "You are a long-context task assistant."
            " Answer strictly based on the provided context."
            " If this is a classification task, choose only from candidate classes."
            " Output only the final answer without extra explanation."
        )

    class_hint = ""
    if isinstance(all_classes, list) and all_classes:
        class_hint = f"\nCandidate classes: {all_classes}\n"

    user_prompt = (
        "Context:\n"
        "----- BEGIN CONTEXT -----\n"
        f"{context}\n"
        "------ END CONTEXT ------\n\n"
        "Task input/question:\n"
        f"{user_input}\n"
        f"{class_hint}\n"
        "Provide final answer only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_messages(
    sample: Dict,
    task: str,
    compressor: Optional[PromptCompressor],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, str]], Dict[str, Optional[float]]]:
    """
    Build chat messages for one sample with structured compression:
    - Keep input/question untouched.
    - Compress only context field.
    """
    context = str(sample.get("context", ""))
    user_input = str(sample.get("input", ""))

    compression_meta = {
        "compression_rate": None,
        "original_context_chars": len(context),
        "compressed_context_chars": len(context),
        "compression_time_seconds": 0.0,
        "compression_error": None,
        "compression_applied": False,
    }

    do_compress = (
        (not args.disable_compression)
        and compressor is not None
        and len(context) >= args.min_context_chars
    )
    if do_compress:
        compressed_context, comp_info = compress_context_structured(
            compressor=compressor,
            context=context,
            question=user_input,
            rate=args.compression_rate,
        )
        context = compressed_context
        compression_meta.update(comp_info)
        compression_meta["compression_applied"] = True

    key = _prompt_task_key(task)
    if key is None:
        sample_with_context = dict(sample)
        sample_with_context["context"] = context
        messages = build_messages_generic(sample_with_context)
        return messages, compression_meta

    template = LONGBENCH_V1_DATASET_PROMPTS[key]
    user_blob = _format_longbench_prompt(template, context, user_input)
    messages = [{"role": "user", "content": user_blob}]
    return messages, compression_meta


def strip_thinking_from_content(text: str) -> str:
    """Drop model thinking blocks; keep only the final user-visible answer."""
    if not text:
        return ""
    out = text
    _t_o = "<" + "think" + ">"
    _t_c = "<" + "/" + "think" + ">"
    _r_o = "<" + "redacted_thinking" + ">"
    _r_c = "<" + "/" + "redacted_thinking" + ">"
    out = re.sub(
        re.escape(_t_o) + r".*?" + re.escape(_t_c) + r"\s*",
        "",
        out,
        flags=re.S | re.I,
    )
    out = re.sub(
        re.escape(_r_o) + r".*?" + re.escape(_r_c) + r"\s*",
        "",
        out,
        flags=re.S | re.I,
    )
    if _t_c in out:
        out = out.split(_t_c, 1)[-1]
    if _r_c in out:
        out = out.split(_r_c, 1)[-1]
    return out.strip()


def infer_one(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> Dict:
    start = time.perf_counter()
    kwargs = {}
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        **kwargs,
    )
    e2e = time.perf_counter() - start
    raw = response.choices[0].message.content or ""
    pred = strip_thinking_from_content(raw).strip()
    return {"pred": pred, "e2e_seconds": round(e2e, 4)}


def run_task(
    client: OpenAI,
    compressor: Optional[PromptCompressor],
    task: str,
    task_path: str,
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    samples = load_jsonl(task_path, max_samples=args.max_samples_per_task)
    out_path = os.path.join(output_dir, f"{task}.jsonl")
    print(f"[task={task}] samples={len(samples)} -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as f_out:
        for sample in tqdm(samples, total=len(samples), desc=task):
            messages, comp_meta = build_messages(
                sample=sample,
                task=task,
                compressor=compressor,
                args=args,
            )
            infer = infer_one(
                client=client,
                model_name=args.model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
            )

            record = {
                "pred": infer["pred"],
                "answers": sample.get("answers", []),
                "all_classes": sample.get("all_classes", None),
                "length": sample.get("length", None),
                "_id": sample.get("_id", None),
                "dataset": sample.get("dataset", task),
                "language": sample.get("language", None),
                "metrics": {
                    "e2e_seconds": infer["e2e_seconds"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "compression_applied": comp_meta["compression_applied"],
                    "compression_rate": comp_meta["compression_rate"],
                    "original_context_chars": comp_meta["original_context_chars"],
                    "compressed_context_chars": comp_meta["compressed_context_chars"],
                    "compression_time_seconds": comp_meta["compression_time_seconds"],
                    "compression_error": comp_meta["compression_error"],
                },
                "raw_input": sample.get("input", ""),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    task2path = list_task_files(args.data_dir)
    tasks = resolve_tasks(args, task2path)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data dir: {args.data_dir}")
    print(f"Selected tasks ({len(tasks)}): {tasks}")
    if args.max_samples_per_task > 0:
        print(f"Max samples per task: {args.max_samples_per_task}")

    compressor = None
    if not args.disable_compression:
        print(f"Loading prompt compressor: {args.compressor_model}")
        compressor = PromptCompressor(
            args.compressor_model,
            use_llmlingua2=True,
        )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    for task in tasks:
        run_task(
            client=client,
            compressor=compressor,
            task=task,
            task_path=task2path[task],
            args=args,
            output_dir=args.output_dir,
        )

    print(f"Done. Per-task result files are in: {args.output_dir}")


if __name__ == "__main__":
    main()
