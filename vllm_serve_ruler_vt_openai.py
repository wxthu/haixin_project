#!/usr/bin/env python3
# 调用 vLLM OpenAI 兼容接口，评估 RULER VT（variable tracking）jsonl，并统计准确率。
# 用法: python vllm_serve_ruler_vt_openai.py path/to/validation.jsonl [-o results.jsonl]

import argparse
import errno
import json
import re
import sys
import time
from typing import Dict, List, Optional, Set

from openai import OpenAI
from tqdm import tqdm


def _flush_result_line(f_out, record: dict) -> None:
    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    f_out.flush()


def _die_on_write_oserror(e: OSError, output_path: str) -> None:
    msg = str(e)
    if e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
        msg = (
            f"写入结果失败（磁盘空间或配额不足）: {output_path}\n"
            f"  系统错误: {e}\n"
            f"  请检查: df -h 与 df -i；已写入的行均保留在文件中，请清理空间后从断点重跑（或换 -o 新路径）。"
        )
    print(f"\n[Fatal] {msg}", file=sys.stderr)
    raise SystemExit(1) from e

# 默认与 vllm_serve_long_context_openai.py 对齐，可用命令行覆盖
DEFAULT_API_KEY = "EMPTY"
DEFAULT_BASE_URL = "http://localhost:8090/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B-AWQ"

MAX_PRED_ITEMS = 32


def build_messages(task_input: str) -> List[Dict[str, str]]:
    system = (
        "你是一个严格按照指令完成信息提取任务的助手。\n"
        "任务文本会要求你找出与某个数值相关的所有变量名。\n"
        "请仅输出变量名列表，使用英文逗号分隔，不要空格或其它说明。\n"
        "变量名保持原文中的大小写形式输出即可。"
    )
    user = (
        "请根据以下任务输入作答，只输出逗号分隔的变量名列表。\n\n"
        f"{task_input.strip()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_var(token: str) -> str:
    token = (token or "").strip()
    token = re.sub(r"^\d+\s*[\.\):：-]\s*", "", token)
    token = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", token)
    return token.upper()


def parse_predicted_set(text: str, max_items: int = MAX_PRED_ITEMS) -> Set[str]:
    raw = (text or "").strip()
    if not raw:
        return set()

    def _clean_item(s: str) -> str:
        return normalize_var(s)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = [_clean_item(str(x)) for x in parsed]
            items = [x for x in items if x]
            return set(items[:max_items])
    except Exception:
        pass

    parts = re.split(r"[,;\n]+", raw)
    cleaned = [_clean_item(p) for p in parts]
    cleaned = [x for x in cleaned if x]
    return set(cleaned[:max_items])


def label_set_from_sample(sample: Dict) -> Set[str]:
    outputs = sample.get("outputs", [])
    if not isinstance(outputs, list):
        return set()
    return {normalize_var(str(x)) for x in outputs if normalize_var(str(x))}


def label_coverage(pred_set: Set[str], gt_set: Set[str]) -> float:
    if not gt_set:
        return 1.0
    return len(pred_set & gt_set) / len(gt_set)


def default_output_path(input_path: str) -> str:
    if input_path.endswith(".jsonl"):
        return input_path[:-6] + "_vt_infer_results.jsonl"
    return input_path + "_vt_infer_results.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="RULER VT jsonl 评测（OpenAI 兼容 vLLM）")
    parser.add_argument(
        "input_jsonl",
        help="VT 任务 validation.jsonl 路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="逐条推理结果输出路径（默认：输入同目录 *_vt_infer_results.jsonl）",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="模型名")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="关闭 enable_thinking（非 Qwen3 思考模型时使用）",
    )
    args = parser.parse_args()

    input_path = args.input_jsonl
    output_path = args.output or default_output_path(input_path)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    line_count = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    print(f"开始处理: {input_path}，共 {line_count} 行 -> 结果: {output_path}")

    n_total = 0
    n_exact = 0
    coverage_sum = 0.0

    extra_body = {} if args.no_thinking else {"enable_thinking": True}

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in tqdm(f_in, total=line_count, desc="VT 推理"):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                task_input = sample.get("input", "")
                gt_set = label_set_from_sample(sample)
                messages = build_messages(task_input)

                start_time = time.perf_counter()
                ttft: Optional[float] = None
                full_content: List[str] = []

                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=args.max_tokens,
                    stream=True,
                    extra_body=extra_body,
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    has_content = getattr(delta, "content", None)
                    has_reasoning = getattr(delta, "reasoning_content", None)
                    if ttft is None and (has_content or has_reasoning):
                        ttft = time.perf_counter() - start_time
                    full_content.append(has_content or "")

                e2e = time.perf_counter() - start_time
                raw_output = "".join(full_content).strip()
                pred_set = parse_predicted_set(raw_output)
                cov = label_coverage(pred_set, gt_set)
                exact = pred_set == gt_set

                rec = {
                    "index": sample.get("index"),
                    "ground_truth_set": sorted(gt_set),
                    "prediction_raw": raw_output,
                    "prediction_set": sorted(pred_set),
                    "intersection": sorted(pred_set & gt_set),
                    "exact_match": exact,
                    "label_coverage": round(cov, 6),
                    "metrics": {
                        "ttft_seconds": round(ttft, 4) if ttft else None,
                        "e2e_seconds": round(e2e, 4),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }
                try:
                    _flush_result_line(f_out, rec)
                except OSError as oe:
                    _die_on_write_oserror(oe, output_path)

                n_total += 1
                if exact:
                    n_exact += 1
                coverage_sum += cov

            except OSError:
                raise
            except Exception as e:
                print(f"\n[Error] 样本失败: {e}")
                continue

    if n_total > 0:
        strict_acc = n_exact / n_total
        avg_cov = coverage_sum / n_total
        file_lines = sum(1 for _ in open(output_path, "r", encoding="utf-8") if _.strip())
        print("\n========== VT 评测汇总 ==========")
        print(f"成功写入样本数: {n_total}（结果文件非空行: {file_lines}）")
        if file_lines != n_total:
            print(f"[警告] 统计与文件行数不一致，请检查 {output_path}")
        print(f"严格准确率（预测集合与标签集合一致）: {strict_acc:.4%}")
        print(f"平均标签召回 |P∩G|/|G|: {avg_cov:.6f}")
        print(f"逐条结果: {output_path}")
    else:
        print("未处理到有效样本。")


if __name__ == "__main__":
    main()
