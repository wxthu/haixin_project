import json
from pathlib import Path
from typing import List, Tuple

from add_label_to_jsonl import extract_answer_text


def load_jsonl(path: str) -> List[dict]:
    data: List[dict] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def keyword_recall_per_sample(labels: List[str], answer: str) -> Tuple[int, int]:
    """
    计算一条样本的关键词“召回”情况：
      - labels: ground truth 抽取出的关键词列表
      - answer: 新模型生成的答案文本

    返回 (matched, total)，即命中的关键词个数和总关键词个数。
    """
    if not labels:
        return 0, 0

    matched = 0
    for kw in labels:
        if not isinstance(kw, str) or not kw:
            continue
        if kw in answer:
            matched += 1
    return matched, len(labels)


def eval_keyword_match(
    gt_with_label_path: str = "summary_results_under_raw_inputs_with_label.jsonl",
    hyp_path: str = "summary_results_under_compressed_inputs.jsonl",
    per_sample_output_path: str = "keyword_match_per_sample.jsonl",
) -> None:
    """
    对比带 label 的 ground truth 与新模型答案，计算每条样本的
    关键词匹配度（召回率），并输出整体平均。

    假设：
      - 两个 JSONL 文件行数一一对应（同一行表示同一个输入）。
      - ground truth 文件中每一行已有 `label` 字段（关键词列表）。
      - 新模型文件中也有 `summary` 字段，结构类似原文件。
    """
    gt_data = load_jsonl(gt_with_label_path)
    hyp_data = load_jsonl(hyp_path)

    if len(gt_data) != len(hyp_data):
        raise ValueError(
            f"Line count mismatch: gt={len(gt_data)}, hyp={len(hyp_data)}. "
            "Ensure the two JSONL files are aligned."
        )

    total_matched = 0
    total_labels = 0

    per_sample_recalls: List[float] = []

    # 打开逐样本结果输出文件
    out_path = Path(per_sample_output_path)
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, (gt_obj, hyp_obj) in enumerate(zip(gt_data, hyp_data)):
            labels = gt_obj.get("label", [])
            if not isinstance(labels, list):
                labels = []

            hyp_summary = hyp_obj.get("summary", "")
            hyp_answer = extract_answer_text(hyp_summary)

            matched, total = keyword_recall_per_sample(labels, hyp_answer)
            if total == 0:
                # 没有关键词时，这条样本不参与整体统计，但仍可输出一条记录
                sample_record = {
                    "index": idx,
                    "matched": 0,
                    "total": 0,
                    "recall": None,
                    "labels": labels,
                    "hyp_answer": hyp_answer,
                }
                fout.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                continue

            recall = matched / total
            total_matched += matched
            total_labels += total
            per_sample_recalls.append(recall)

            sample_record = {
                "index": idx,
                "matched": matched,
                "total": total,
                "recall": recall,
                "labels": labels,
                "hyp_answer": hyp_answer,
            }
            fout.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

    if total_labels == 0:
        print("No labels found in ground truth file; nothing to evaluate.")
        print(f"Per-sample details are still written to: {out_path}")
        return

    macro_recall = sum(per_sample_recalls) / len(per_sample_recalls)
    micro_recall = total_matched / total_labels

    print(f"Samples with labels: {len(per_sample_recalls)}")
    print(f"Total labels      : {total_labels}")
    print(f"Matched labels    : {total_matched}")
    print(f"Macro recall      : {macro_recall:.4f}")
    print(f"Micro recall      : {micro_recall:.4f}")
    print(f"Per-sample details written to: {out_path}")


if __name__ == "__main__":
    eval_keyword_match()

