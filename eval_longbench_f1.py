#!/usr/bin/env python3
"""
Evaluate LongBench-style JSONL outputs.

- Most tasks: token-level F1 (SQuAD-style); per sample, max over answers[].
- dureader: ROUGE-L F-measure via `rouge-score` (char tokenizer for Chinese), max over answers[].

Requires: `pip install rouge-score` (import name `rouge_score`) when evaluating dureader.

Report: one table; task column shows name, sample count, and [rougeL] for dureader.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Folder basename -> short label (column order follows this dict's insertion order)
METHOD_DIR_TO_LABEL = {
    "longbench_v1_results": "raw",
    "longbench_v1_results_compressed": "llmlingua",
    "longbench_v1_results_elbow": "elbow_one",
    "longbench_v1_results_new_elbow": "elbow_two",
    "longbench_v1_results_mass_75": "mass_75",
}

# Task key -> jsonl filename; row order follows this list
TASK_ORDER = [
    "hotpotqa",
    "narrativeqa",
    "multifieldqa_en",
    "multifieldqa_zh",
    "qasper",
    "2wikimqa",
    "dureader",
]

TASK_TO_JSONL: Dict[str, str] = {t: f"{t}.jsonl" for t in TASK_ORDER}

# Tasks evaluated with ROUGE-L (char-level) instead of token F1
TASK_ROUGE_L: frozenset[str] = frozenset({"dureader"})

_PUNCT_EXTRA = "。，、；：？！「」『』【】《》〈〉（）［］〔〕…—·％"


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation, remove articles, collapse whitespace (LongBench/SQuAD + CJK punct)."""

    def lower(t: str) -> str:
        return t.lower()

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        exclude.update(_PUNCT_EXTRA)
        return "".join(ch for ch in t if ch not in exclude)

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _f1_multiset(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    overlap = sum((pred_c & gold_c).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(pred_c.values())
    rec = overlap / sum(gold_c.values())
    return 2 * prec * rec / (prec + rec)


def f1_for_pred_and_golds(pred: str, golds: Iterable[str]) -> float:
    pred_n = normalize_answer(pred or "")
    pred_tokens = pred_n.split()
    best = 0.0
    for g in golds:
        if not g:
            continue
        gold_tokens = normalize_answer(str(g)).split()
        best = max(best, _f1_multiset(pred_tokens, gold_tokens))
    return best


_ROUGE_L_SCORER: Any = None


def _get_rouge_l_scorer():
    """Lazy singleton; raises SystemExit with install hint if package missing."""
    global _ROUGE_L_SCORER
    if _ROUGE_L_SCORER is not None:
        return _ROUGE_L_SCORER
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise SystemExit(
            "dureader 需要安装 rouge-score 包（PyPI 名带连字符）:\n"
            "  pip install rouge-score"
        ) from e

    class CharTokenizer:
        def tokenize(self, text: str):
            return list(text)

    _ROUGE_L_SCORER = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=CharTokenizer()
    )
    return _ROUGE_L_SCORER


def rouge_l_fmeasure(pred: str, ref: str) -> float:
    """ROUGE-L F-measure: reference = ref, candidate = pred (char-level tokenizer)."""
    ref_s = (ref or "").strip()
    pred_s = (pred or "").strip()
    if not ref_s and not pred_s:
        return 1.0
    if not ref_s or not pred_s:
        return 0.0
    s = _get_rouge_l_scorer().score(ref_s, pred_s)
    return float(s["rougeL"].fmeasure)


def rouge_l_for_pred_and_golds(pred: str, golds: Iterable[str]) -> float:
    best = 0.0
    for g in golds:
        if not g:
            continue
        best = max(best, rouge_l_fmeasure(str(pred), str(g)))
    return best


def _iter_jsonl_pred_answers(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj.get("pred", "")
            answers = obj.get("answers") or []
            if not isinstance(answers, list):
                answers = [answers]
            golds = [str(a) for a in answers if a is not None and str(a).strip()]
            yield str(pred), golds


def eval_jsonl_f1(path: Path) -> Tuple[int, float]:
    """Return (num_samples, mean token F1)."""
    scores: List[float] = []
    for pred, golds in _iter_jsonl_pred_answers(path):
        if not golds:
            scores.append(0.0)
        else:
            scores.append(f1_for_pred_and_golds(pred, golds))
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    return n, mean


def eval_jsonl_rouge_l(path: Path) -> Tuple[int, float]:
    """Return (num_samples, mean ROUGE-L F1, char-level)."""
    scores: List[float] = []
    for pred, golds in _iter_jsonl_pred_answers(path):
        if not golds:
            scores.append(0.0)
        else:
            scores.append(rouge_l_for_pred_and_golds(pred, golds))
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    return n, mean


def _task_label_with_n(task: str, scores: Dict[str, Dict[str, Tuple[int, float]]]) -> str:
    """task name + sample count; if counts differ across methods, show min-max."""
    ns = sorted({t[0] for t in scores.get(task, {}).values()})
    if not ns:
        base = f"{task} (?)"
    elif len(ns) == 1:
        base = f"{task} ({ns[0]})"
    else:
        base = f"{task} ({ns[0]}-{ns[-1]})"
    if task in TASK_ROUGE_L:
        return f"{base} [rougeL]"
    return base


def build_f1_table(
    tasks: List[str],
    method_labels: List[str],
    scores: Dict[str, Dict[str, Tuple[int, float]]],
) -> str:
    """scores[task][method] -> (n, f1)."""
    headers = ["task"] + method_labels
    str_rows: List[List[str]] = []
    for task in tasks:
        row: List[str] = [_task_label_with_n(task, scores)]
        for m in method_labels:
            cell = scores.get(task, {}).get(m)
            if cell is None:
                row.append("-")
            else:
                _, f1 = cell
                row.append(f"{f1:.6f}")
        str_rows.append(row)

    col_widths: List[int] = []
    for col_idx in range(len(headers)):
        w = len(headers[col_idx])
        for r in str_rows:
            w = max(w, len(r[col_idx]))
        col_widths.append(w)

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def parse_tasks(arg: str) -> List[str]:
    if not arg.strip():
        return list(TASK_ORDER)
    names = [t.strip() for t in arg.split(",") if t.strip()]
    bad = [t for t in names if t not in TASK_TO_JSONL]
    if bad:
        raise SystemExit(f"Unknown task(s): {bad}. Known: {TASK_ORDER}")
    return names


def collect_scores(
    root: Path,
    tasks: List[str],
    ordered_dirs: List[str],
) -> Dict[str, Dict[str, Tuple[int, float]]]:
    out: Dict[str, Dict[str, Tuple[int, float]]] = {t: {} for t in tasks}
    for task in tasks:
        jsonl_name = TASK_TO_JSONL[task]
        for dir_name in ordered_dirs:
            label = METHOD_DIR_TO_LABEL[dir_name]
            path = root / dir_name / jsonl_name
            if not path.is_file():
                print(f"[skip] missing {path}", file=sys.stderr)
                continue
            if task in TASK_ROUGE_L:
                n, mean_v = eval_jsonl_rouge_l(path)
            else:
                n, mean_v = eval_jsonl_f1(path)
            out[task][label] = (n, mean_v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean token F1: tasks (rows) x methods (columns)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root containing longbench_v1_results* dirs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report here (default: <root>/longbench_f1_eval.txt)",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help=f"Comma-separated subset of tasks; default = all ({','.join(TASK_ORDER)})",
    )
    args = parser.parse_args()
    root: Path = args.root
    out_path = args.output or (root / "longbench_f1_eval.txt")
    tasks = parse_tasks(args.tasks)

    ordered_dirs = list(METHOD_DIR_TO_LABEL.keys())
    method_labels = [METHOD_DIR_TO_LABEL[d] for d in ordered_dirs]

    scores = collect_scores(root, tasks, ordered_dirs)
    table_f1 = build_f1_table(tasks, method_labels, scores)

    header_txt = (
        "metrics: token_f1 (SQuAD multiset, normalized; max over answers[]) for most tasks; "
        "dureader = ROUGE-L F1 via rouge-score (char tokenizer; max over answers[])\n"
        f"root={root}\n\n"
    )
    report = header_txt + table_f1 + "\n"
    out_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
