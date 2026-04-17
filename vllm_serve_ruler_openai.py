import json
import re
import time
from typing import Dict, List, Set

import requests
from tqdm import tqdm


# --- 基础配置（根据需要自行修改） ---
BASE_URL = "http://localhost:8090"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
MAX_PRED_ITEMS = 10

# RULER 本地 jsonl 文件路径（每行一个样本）
# 每条数据需包含字段：
#   "index", "input", "outputs"
INPUT_FILE = "ruler_dataset.jsonl"

# 预测结果保存文件
OUTPUT_FILE = "ruler_results.jsonl"


def build_prompt(sample: Dict) -> str:
    """
    将 RULER 样本转成对话格式。
    期望模型输出若干词项，使用逗号分隔。
    """
    task_input = sample.get("input", "")

    system_prompt = (
        "你是一个严格按照指令完成信息提取任务的助手。\n"
        "你会看到一个任务输入，其中要求找出最常出现的词。\n"
        "请仅输出你预测的词列表，使用英文逗号分隔。\n"
        "不要输出解释、序号或多余文本。"
    )

    user_prompt = (
        "请根据以下任务输入进行回答。\n"
        "你只需要输出词列表，用英文逗号分隔。\n\n"
        f"{task_input}\n\n"
        "请直接输出结果。"
    )

    return f"{system_prompt}\n\n{user_prompt}"


def call_vllm_api_server(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """
    调用 vllm.entrypoints.api_server 的 /generate 接口。
    """
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        if isinstance(data.get("text"), list) and data["text"]:
            text = str(data["text"][0]).strip()
            # vllm api_server 的 /generate 常返回 prompt + completion，这里去掉 prompt 前缀
            if text.startswith(prompt):
                text = text[len(prompt) :].strip()
            return text
        if isinstance(data.get("generated_text"), str):
            text = data["generated_text"].strip()
            if text.startswith(prompt):
                text = text[len(prompt) :].strip()
            return text
    return str(data).strip()


def normalize_word(token: str) -> str:
    token = (token or "").strip().lower()
    # 去掉首尾常见标点，保留词中连字符
    token = re.sub(r"^[^a-z0-9]+|[^a-z0-9-]+$", "", token)
    return token


def parse_predicted_set(text: str, max_items: int = MAX_PRED_ITEMS) -> Set[str]:
    """
    从模型输出中解析词集合，支持以下格式：
    1) JSON 列表: ["a", "b", ...]
    2) 逗号/换行/分号分隔文本
    """
    raw = (text or "").strip()
    if not raw:
        return set()

    def _clean_item(s: str) -> str:
        s = (s or "").strip()
        # 去掉常见编号前缀，如 "1. xxx" / "2) xxx"
        s = re.sub(r"^\d+\s*[\.\):：-]\s*", "", s)
        return normalize_word(s)

    # 尝试解析 JSON 列表
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
    return {normalize_word(str(x)) for x in outputs if normalize_word(str(x))}


def label_coverage_score(pred_set: Set[str], gt_set: Set[str]) -> float:
    """
    按标签集合归一化的命中率：
    score = |pred ∩ gt| / |gt|
    与 |pred| 无关。
    """
    if not gt_set:
        return 1.0
    inter = pred_set & gt_set
    return len(inter) / len(gt_set)


def process_ruler():
    line_count = sum(1 for _ in open(INPUT_FILE, "r", encoding="utf-8"))
    print(f"开始评估 RULER 数据集: {INPUT_FILE}，共 {line_count} 条样本...")

    n_total = 0
    score_sum = 0.0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as f_out:
        for line in tqdm(f_in, total=line_count, desc="模型推理中"):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                gt_set = label_set_from_sample(sample)
                prompt = build_prompt(sample)

                start_time = time.perf_counter()
                # api_server /generate 非流式，无法直接获取 TTFT
                ttft = None
                raw_output = call_vllm_api_server(
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.0,
                )
                e2e_time = time.perf_counter() - start_time

                pred_set = parse_predicted_set(raw_output)
                score = label_coverage_score(pred_set, gt_set)

                n_total += 1
                score_sum += score

                result_item = {
                    "index": sample.get("index", None),
                    "ground_truth_set": sorted(gt_set),
                    "prediction_raw": raw_output,
                    "prediction_set": sorted(pred_set),
                    "set_intersection": sorted(pred_set & gt_set),
                    "ground_truth_size": len(gt_set),
                    "label_coverage": round(score, 6),
                    "metrics": {
                        "ttft_seconds": round(ttft, 4) if ttft is not None else None,
                        "e2e_seconds": round(e2e_time, 4),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }

                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\n[Error] 处理样本失败: {e}")
                continue


    if n_total > 0:
        avg_score = score_sum / n_total
        print(f"\n评估完成！总样本数: {n_total}, 平均 Label Coverage: {avg_score:.6f}")
    else:
        print("\n评估完成，但未能统计到有效样本。")

    print(f"详细结果已写入：{OUTPUT_FILE}")


if __name__ == "__main__":
    process_ruler()
