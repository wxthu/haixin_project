#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_strategy(filename: str) -> str:
    prefix = "validation_vt_"
    suffix = "_infer_results.jsonl"
    if filename == "validation_vt_infer_results.jsonl":
        return "raw"
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix) : -len(suffix)]
    return ""


def compute_vt_score(file_path: Path) -> Tuple[int, float]:
    total = 0
    score_sum = 0.0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            intersection = item.get("intersection", [])
            ground_truth_set = item.get("ground_truth_set", [])

            inter_len = len(intersection) if isinstance(intersection, list) else 0
            gt_len = len(ground_truth_set) if isinstance(ground_truth_set, list) else 0
            per_item_score = (inter_len / gt_len) if gt_len > 0 else 0.0

            total += 1
            score_sum += per_item_score

    avg_score = (score_sum / total) if total else 0.0
    return total, avg_score


def build_table(
    lengths: List[str],
    strategies: List[str],
    results: Dict[str, Dict[str, Tuple[int, float]]],
) -> str:
    headers = ["maxseq"] + strategies
    rows: List[List[str]] = []

    for length in lengths:
        row = [length]
        for strategy in strategies:
            cell = "-"
            if length in results and strategy in results[length]:
                _, avg_score = results[length][strategy]
                cell = f"{avg_score:.4f}"
            row.append(cell)
        rows.append(row)

    col_widths: List[int] = []
    for col_idx in range(len(headers)):
        max_len = len(headers[col_idx])
        for row in rows:
            max_len = max(max_len, len(row[col_idx]))
        col_widths.append(max_len)

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    maxseq_dirs = sorted(root.glob("maxseq_*"), key=lambda p: int(p.name.split("_", 1)[1]))

    results: Dict[str, Dict[str, Tuple[int, float]]] = {}
    all_strategies = set()

    for maxseq_dir in maxseq_dirs:
        length = maxseq_dir.name.replace("maxseq_", "")
        vt_dir = maxseq_dir / "vt"
        if not vt_dir.exists():
            continue

        for file_path in sorted(vt_dir.glob("validation_vt*_infer_results.jsonl")):
            strategy = parse_strategy(file_path.name)
            if not strategy:
                continue
            total, avg_score = compute_vt_score(file_path)
            results.setdefault(length, {})[strategy] = (total, avg_score)
            all_strategies.add(strategy)

    strategy_order = ["raw", "llmlingua", "llmlingua_elbow", "llmlingua_new_elbow", "llmlingua_mass_80"]
    remaining = sorted(s for s in all_strategies if s not in strategy_order)
    strategies = [s for s in strategy_order if s in all_strategies] + remaining
    lengths = sorted(results.keys(), key=lambda x: int(x))

    table = build_table(lengths, strategies, results)
    output_path = root / "vt_accuracy_summary.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("VT accuracy summary by max sequence length and strategy\n")
        f.write("score(item) = len(intersection) / len(ground_truth_set)\n")
        f.write("table value = average score over all items in file\n\n")
        f.write(table)
        f.write("\n")

    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()
