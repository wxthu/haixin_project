"""
将 conversations[*].value 拆成「不可压缩」与「可压缩」段，对可压缩段调用 compress() 后再按序拼接。

约定结构（与 train_data_summary 系列一致）:
  ### ... 任务说明 ... ###\\n
  <用户问题，可多行>
  [SEP]
  0.xxx
  1.xxx
  ...
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Callable, List, NamedTuple

SEP = "[SEP]"
DOC_START = re.compile(r"^\d+\.", re.MULTILINE)

__all__ = [
    "Segment",
    "split_documents",
    "parse_conversation_value",
    "compress_conversation_value",
    "default_compress",
]


class Segment(NamedTuple):
    text: str
    compressible: bool


def split_documents(post_sep: str) -> List[str]:
    """按行首 '数字.' 切分检索文档；首段若无换行前缀也可匹配（如 '[SEP]0.xxx'）。"""
    post_sep = post_sep.lstrip("\n")
    if not post_sep.strip():
        return []

    matches = list(DOC_START.finditer(post_sep))
    if not matches:
        return [post_sep]

    chunks: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(post_sep)
        chunks.append(post_sep[start:end])
    return chunks


def parse_conversation_value(value: str) -> List[Segment]:
    """
    将 value 拆成顺序片段。
    - 不可压缩: 指令首行 + 换行, 用户问题, 分隔符 [SEP]
    - 可压缩: 每一条编号文档（保持原文顺序与编号，便于与 chosen/rejected 对齐）
    """
    if SEP not in value:
        return [Segment(value, False)]

    pre_sep, post_sep = value.split(SEP, 1)

    if "\n" in pre_sep:
        header, query = pre_sep.split("\n", 1)
        header = header + "\n"
    else:
        header, query = pre_sep, ""

    segments: List[Segment] = [
        Segment(header, False),
        Segment(query, False),
        Segment(SEP, False),
    ]

    for doc in split_documents(post_sep):
        if doc:
            segments.append(Segment(doc, True))

    return segments


def compress_conversation_value(
    value: str,
    compress: Callable[[str], str],
) -> str:
    """对 parse 结果逐段处理，可压缩段走 compress。"""
    parts: List[str] = []
    for seg in parse_conversation_value(value):
        if seg.compressible:
            parts.append(compress(seg.text))
        else:
            parts.append(seg.text)
    return "".join(parts)


def default_compress(text: str) -> str:
    """占位：替换为你的 compress(query: str) 实现。"""
    return text


def load_json_samples(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="解析并压缩 conversation value")
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="可选：ORPO JSON 路径，打印第一条 before/after（不调用外部压缩，仅演示拼接）",
    )
    args = parser.parse_args()

    if args.json:
        data = load_json_samples(args.json)
        raw = data[0]["conversations"][0]["value"]
        segs = parse_conversation_value(raw)
        print("segments:", len(segs))
        for i, s in enumerate(segs):
            kind = "compressible" if s.compressible else "fixed"
            preview = s.text[:120].replace("\n", "\\n")
            print(f"  [{i}] {kind} len={len(s.text)} preview={preview!r}...")
        out = compress_conversation_value(raw, default_compress)
        print("same_as_input:", out == raw)
        return

    demo = (
        "### 相关媒资提取，输出格式：文档序号#名称|类型|相关系数 ###\n"
        "示例问题？[SEP]"
        "0.短文档A\t类型：电影\n"
        "1.非常长的文档B..." + "x" * 200
    )
    segs = parse_conversation_value(demo)
    assert len(segs) == 5  # header, query, SEP, doc0, doc1
    print("demo segments:", [(s.compressible, len(s.text)) for s in segs])


if __name__ == "__main__":
    main()
