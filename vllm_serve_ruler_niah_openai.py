#!/usr/bin/env python3
# 调用 vLLM OpenAI 兼容接口（需 vllm.entrypoints.openai.api_server），评估 RULER NIAH single jsonl。
# 用法: python vllm_serve_ruler_niah_openai.py path/to/validation.jsonl [-o results.jsonl]

import argparse
import errno
import json
import re
import sys
import time
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


def _flush_result_line(f_out, record: dict) -> None:
    """写入一行结果并 flush；磁盘错误向上抛出，便于立刻停止，避免统计与文件不一致。"""
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

DEFAULT_API_KEY = "EMPTY"
DEFAULT_BASE_URL = "http://localhost:8091/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B-AWQ"


def build_messages(task_input: str) -> List[Dict[str, str]]:
    system = (
        "你是一个严格按照指令完成信息提取任务的助手。\n"
        "任务会要求你从长文本中找出给定的「魔法数字」（magic number）。\n"
        "请只输出该数字本身，不要输出其它文字、标点或解释。"
    )
    user = (
        "请根据以下任务输入作答，只输出数字。\n\n"
        f"{task_input.strip()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ground_truth_number(sample: Dict) -> str:
    out = sample.get("outputs")
    if not isinstance(out, list) or not out:
        return ""
    return str(out[0]).strip()


def parse_predicted_number(text: str, gt: str) -> str:
    """
    从模型输出中解析目标数字。优先与标签一致；否则按数字串启发式提取。
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    # 整段即为数字（允许末尾句号）
    compact = raw.rstrip(".").strip()
    if compact.isdigit():
        return compact

    nums = re.findall(r"\d+", raw)
    if not nums:
        return ""
    if gt and gt in nums:
        return gt
    if len(nums) == 1:
        return nums[0]
    gl = len(gt) if gt else 0
    if gl > 0:
        same_len = [n for n in nums if len(n) == gl]
        if len(same_len) == 1:
            return same_len[0]
    return nums[-1]


def default_output_path(input_path: str) -> str:
    if input_path.endswith(".jsonl"):
        return input_path[:-6] + "_niah_infer_results.jsonl"
    return input_path + "_niah_infer_results.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="RULER NIAH single jsonl 评测（OpenAI 兼容 vLLM）")
    parser.add_argument("input_jsonl", help="NIAH 任务 validation.jsonl 路径")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="逐条推理结果输出路径（默认：输入同目录 *_niah_infer_results.jsonl）",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
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
    n_correct = 0
    extra_body = {} if args.no_thinking else {"enable_thinking": True}

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in tqdm(f_in, total=line_count, desc="NIAH 推理"):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                task_input = sample.get("input", "")
                gt = ground_truth_number(sample)
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
                pred = parse_predicted_number(raw_output, gt)
                exact = bool(gt) and pred == gt

                rec = {
                    "index": sample.get("index"),
                    "ground_truth": gt,
                    "prediction_raw": raw_output,
                    "prediction_parsed": pred,
                    "exact_match": exact,
                    "token_position_answer": sample.get("token_position_answer"),
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
                    n_correct += 1

            except OSError:
                raise
            except Exception as e:
                print(f"\n[Error] 样本失败: {e}")
                continue

    if n_total > 0:
        acc = n_correct / n_total
        file_lines = sum(1 for _ in open(output_path, "r", encoding="utf-8") if _.strip())
        print("\n========== NIAH 评测汇总 ==========")
        print(f"成功写入样本数: {n_total}（结果文件非空行: {file_lines}）")
        if file_lines != n_total:
            print(f"[警告] 统计与文件行数不一致，请检查 {output_path}")
        print(f"准确率（解析后数字与标签一致）: {acc:.4%}")
        print(f"逐条结果: {output_path}")
    else:
        print("未处理到有效样本。")


if __name__ == "__main__":
    main()
