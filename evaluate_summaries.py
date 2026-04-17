import argparse
import json
import re
from typing import List, Tuple

from rouge_score import rouge_scorer
from bert_score import score as bertscore


def load_summaries(
    path_ref: str,
    path_hyp: str,
) -> Tuple[List[str], List[str]]:
    """
    读取两份 jsonl 文件，按行对齐抽取 summary。
    默认把 <think>...</think> 思维链去掉，只保留对用户可见的总结内容。
    """
    refs: List[str] = []
    hyps: List[str] = []

    with open(path_ref, "r", encoding="utf-8") as f_ref, open(
        path_hyp, "r", encoding="utf-8"
    ) as f_hyp:
        for i, (l_ref, l_hyp) in enumerate(zip(f_ref, f_hyp), start=1):
            l_ref = l_ref.strip()
            l_hyp = l_hyp.strip()
            if not l_ref or not l_hyp:
                continue
            try:
                j_ref = json.loads(l_ref)
                j_hyp = json.loads(l_hyp)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败（第 {i} 行）: {e}") from e

            s_ref = j_ref.get("summary", "")
            s_hyp = j_hyp.get("summary", "")

            def strip_think(x: str) -> str:
                # 去掉 <think>...</think>，只保留后面的可见回答
                if "<think>" in x and "</think>" in x:
                    # 贪婪匹配整段思维链
                    x = re.sub(r"<think>.*?</think>\s*", "", x, flags=re.S)
                return x.strip()

            s_ref = strip_think(s_ref)
            s_hyp = strip_think(s_hyp)

            refs.append(s_ref)
            hyps.append(s_hyp)

    # 确保行数一致（如果 hyp 少于 ref，多余的 ref 会被截断；如果多于，同样被 zip 截断）
    if not refs or not hyps:
        raise ValueError("未从文件中成功解析到任何 summary。")

    return refs, hyps


def compute_rouge(
    refs: List[str],
    hyps: List[str],
    tokenizer: "object | None" = None,
) -> dict:
    """
    计算 ROUGE-1 / ROUGE-2 / ROUGE-L（F1），并返回总体平均。

    说明：
    - 对纯中文文本，rouge_score 默认的英文分词器几乎提取不到 token，
      会导致分数异常为 0。这里支持传入自定义 tokenizer，
      默认使用简单的“按字符切分”方案，更适合中文。
    """
    if len(refs) != len(hyps):
        raise ValueError(
            f"refs 与 hyps 数量不一致: {len(refs)} vs {len(hyps)}（应逐条对齐）"
        )

    # 默认使用字符级 tokenizer，更友好地支持中文
    if tokenizer is None:
        class CharTokenizer:
            def tokenize(self, text: str):
                return list(text)

        tokenizer = CharTokenizer()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=tokenizer,
    )

    total_r1 = total_r2 = total_rl = 0.0
    n = len(refs)

    for r, h in zip(refs, hyps):
        scores = scorer.score(r, h)
        total_r1 += scores["rouge1"].fmeasure
        total_r2 += scores["rouge2"].fmeasure
        total_rl += scores["rougeL"].fmeasure

    return {
        "rouge1_f": total_r1 / n,
        "rouge2_f": total_r2 / n,
        "rougeL_f": total_rl / n,
        "count": n,
    }


def compute_bertscore(
    refs: List[str],
    hyps: List[str],
    lang: str = "zh",
) -> dict:
    """
    使用 BERTScore 计算语义相似度（P/R/F1 均值）。
    说明：一次性 batch 计算，避免逐条跑模型。
    """
    # bertscore.score 接受 (cands, refs)
    P, R, F1 = bertscore(
        cands=hyps,
        refs=refs,
        lang=lang,
        rescale_with_baseline=False,
    )

    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
        "count": len(refs),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "对比两份 summary jsonl（raw 作为参考，compressed 作为候选），"
            "计算 ROUGE-1/2/L 和 BERTScore。"
        )
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="summary_results_under_raw_inputs.jsonl",
        help="参考摘要文件（标准），jsonl 格式，默认当前目录下的 summary_results_under_raw_inputs.jsonl",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="summary_results_under_compressed_inputs.jsonl",
        help="候选摘要文件（压缩输入生成结果），jsonl 格式，默认当前目录下的 summary_results_under_compressed_inputs.jsonl",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        help="BERTScore 语言代码（默认 zh）",
    )
    parser.add_argument(
        "--rouge_tokenizer",
        type=str,
        choices=["char", "default"],
        default="char",
        help=(
            "ROUGE 分词方式：char=字符级（推荐中文，默认），"
            "default=rouge_score 自带英文分词"
        ),
    )

    args = parser.parse_args()

    refs, hyps = load_summaries(args.ref, args.hyp)

    print(f"共加载样本: {len(refs)} 条\n")

    # ROUGE
    rouge_tokenizer = None
    if args.rouge_tokenizer == "default":
        rouge_tokenizer = None  # 使用 rouge_score 默认分词
    else:
        # 显式用字符级分词（与 compute_rouge 默认行为一致，便于将来修改）
        class CharTokenizer:
            def tokenize(self, text: str):
                return list(text)

        rouge_tokenizer = CharTokenizer()

    rouge_stats = compute_rouge(refs, hyps, tokenizer=rouge_tokenizer)
    print("===== ROUGE (平均 F1) =====")
    print(f"ROUGE-1 F: {rouge_stats['rouge1_f']:.4f}")
    print(f"ROUGE-2 F: {rouge_stats['rouge2_f']:.4f}")
    print(f"ROUGE-L F: {rouge_stats['rougeL_f']:.4f}")
    print()

    # BERTScore
    bert_stats = compute_bertscore(refs, hyps, lang=args.lang)
    print("===== BERTScore (平均) =====")
    print(f"P:  {bert_stats['bertscore_precision']:.4f}")
    print(f"R:  {bert_stats['bertscore_recall']:.4f}")
    print(f"F1: {bert_stats['bertscore_f1']:.4f}")


if __name__ == "__main__":
    main()

