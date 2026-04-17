import json
import re
from pathlib import Path
from typing import List

try:
    import jieba.posseg as pseg
except ImportError:
    pseg = None  # 用户需要自行安装 jieba


def extract_answer_text(summary_field: str) -> str:
    """
    Given the `summary` field from JSONL, strip the <think>...</think> 部分，
    只保留真正给用户看的回答文本。
    """
    if not isinstance(summary_field, str):
        return ""
    if "</think>" in summary_field:
        return summary_field.split("</think>", 1)[-1].strip()
    return summary_field.strip()


def extract_keywords_from_answer(answer: str) -> List[str]:
    """
    从最终回答中抽取较重要的关键词，作为 label。

    设计目标：简单、干净、稳定。
    仅保留三类信息：
      - 片名 / 作品名：所有出现在《》中的内容
      - 完整日期：形如 2019年4月24日，统一成 2019-04-24
      - 人名：使用 jieba 词性标注（flag == 'nr'）抽取 2~4 字中文人名

    返回去重后的字符串列表。
    """
    if not answer:
        return []

    labels: List[str] = []

    # 1) 作品名：所有 《...》 中的内容
    title_pattern = re.compile(r"《([^》]+)》")
    titles = title_pattern.findall(answer)
    labels.extend(titles)

    # 2) 完整日期（年-月-日），统一成 'YYYY-MM-DD' 形式，便于后续分析
    date_pattern = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?")

    def norm_date(y: str, m: str, d: str) -> str:
        return f"{y}-{int(m):02d}-{int(d):02d}"

    for y, m, d in date_pattern.findall(answer):
        labels.append(norm_date(y, m, d))

    # 3) 人名：如果安装了 jieba，则用词性标注抽取 nr
    if pseg is not None:
        try:
            for w in pseg.cut(answer):
                # 只要人名（nr），长度在 2~4 个汉字之间
                if (
                    w.flag == "nr"
                    and 2 <= len(w.word) <= 4
                    and all("\u4e00" <= ch <= "\u9fff" for ch in w.word)
                ):
                    labels.append(w.word)
        except Exception:
            # 分词异常时忽略人名，不影响其它 label
            pass

    # 去重并保持顺序
    seen = set()
    deduped: List[str] = []
    for k in labels:
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        deduped.append(k)

    return deduped


def add_label_to_jsonl(
    input_path: str = "summary_results_under_raw_inputs.jsonl",
    output_path: str = "summary_results_under_raw_inputs_with_label.jsonl",
) -> None:
    """
    为 JSONL 中每条样本自动生成一个 `label` 字段：
      - 从 `summary` 里抽取最终回答文本
      - 用启发式规则抽取关键词列表，作为 `label`

    输出为新的 JSONL 文件，不会覆盖原始 ground truth。
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            summary_field = obj.get("summary", "")
            answer = extract_answer_text(summary_field)
            labels = extract_keywords_from_answer(answer)

            obj["label"] = labels
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    add_label_to_jsonl()
