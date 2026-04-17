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
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

SEP = "[SEP]"
#
# 文档块起点判定（比单纯 `^\\d+\\.` 更严格）：
# - 避免在 `网页内容` 等字段中出现的“内部编号行”被误判为新的文档块
# - 文档块起点通常在同一行中包含 `标题：` 或 `网页标题：` 或 `类型：`
DOC_START = re.compile(r"^(?P<idx>\d+)\..*(?:标题：|网页标题：|类型：)", re.MULTILINE)
DOC_IDX_PREFIX = re.compile(r"^(?P<idx>\d+)\.", re.MULTILINE)

__all__ = [
    "Segment",
    "split_documents",
    "parse_conversation_value",
    "compress_conversation_value",
    "compress_document_block",
    "default_compress",
    "CompressionConfig",
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


@dataclass(frozen=True)
class CompressionConfig:
    """
    [SEP] 后文档块的“结构不变压缩”策略。

    约定：
    - 保留文档块开头的编号（如 `0.` / `12.`）
    - 保留字段名（`标题：` / `网页内容：` 等）
    - 对字段值做截断（仍使用制表符 `\\t` 作为字段分隔），以减少 prompt 长度
    """

    max_first_segment_len: Optional[int] = 500
    default_max_value_len: int = 200
    value_trunc_len_by_key_contains: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.value_trunc_len_by_key_contains is None:
            object.__setattr__(
                self,
                "value_trunc_len_by_key_contains",
                {
                    # 页面类字段通常占用最大
                    "网页内容": 250,
                    # 结构化摘要字段
                    "剧情简介": 220,
                    "简介": 160,
                    "看点": 130,
                    "标签": 110,
                    "主要演员": 120,
                    "角色": 120,
                    "别名": 180,
                },
            )


def _truncate(text: str, max_len: int) -> str:
    """
    截断文本，但保留末尾的空白字符（通常包含文档块分隔用的换行 `\\n`）。
    这样能避免“截掉末尾 \\n 导致下一块编号不再是行首”的问题。
    """
    if len(text) <= max_len:
        return text

    m = re.search(r"[\s\r\n]*$", text)
    tail = m.group(0) if m else ""
    core = text[: len(text) - len(tail)] if tail else text

    if len(core) <= max_len:
        return core + tail
    return core[:max_len] + "..." + tail


def _maybe_truncate_by_field_key(field_key: str, field_value: str, cfg: CompressionConfig) -> str:
    """
    根据字段名片段（contains）决定截断长度。
    key 与 value 都来自原文，尽量保持结构与字段分隔不变。
    """
    key = field_key.strip()
    max_len = cfg.default_max_value_len
    for k_contains, v_max in cfg.value_trunc_len_by_key_contains.items():
        if k_contains in key:
            max_len = v_max
            break
    return _truncate(field_value, max_len)


def compress_document_block(text: str, cfg: Optional[CompressionConfig] = None) -> str:
    """
    压缩单个 `[SEP]` 后的“编号文档块”，但保持字段结构不变：
    - `idx.` 前缀原样保留
    - tab 分隔的字段顺序原样保留
    - 字段名不变；字段值做截断

    注意：这是一个“确定性/无模型”的压缩策略，便于测试与复现。
    """
    if cfg is None:
        cfg = CompressionConfig()

    # 文档块本身是一个大字符串，字段大多以 tab 分隔
    parts = text.split("\t")
    if not parts:
        return text

    # 第一段包含 `idx.` 与可能的第一个字段（例如 `0.标题：...` / `8.网页标题：...`）
    first = parts[0]
    if cfg.max_first_segment_len is not None:
        # 只做“轻量”截断，避免把标题/编号破坏掉
        first = _truncate(first, cfg.max_first_segment_len)

    new_parts: List[str] = [first]

    for seg in parts[1:]:
        # route 字段：常见形如 `retrieve route:ES`，不使用中文全角冒号分隔字段
        if "：" not in seg:
            new_parts.append(seg)
            continue

        field_key, field_value = seg.split("：", 1)
        truncated_value = _maybe_truncate_by_field_key(field_key, field_value, cfg)
        new_parts.append(field_key + "：" + truncated_value)

    return "\t".join(new_parts)


def default_compress(text: str) -> str:
    """默认压缩函数：只压缩编号文档块（[SEP] 之后），保留结构与字段名。"""
    return compress_document_block(text)


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
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="当 --json 指定时，选择第几个样本（conversations[0] 固定使用）。",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="仅解析并打印分段，不对 [SEP] 后做压缩。",
    )
    args = parser.parse_args()

    if args.json:
        data = load_json_samples(args.json)
        raw = data[args.index]["conversations"][0]["value"]
        segs = parse_conversation_value(raw)
        print("segments:", len(segs))
        for i, s in enumerate(segs):
            kind = "compressible" if s.compressible else "fixed"
            preview = s.text[:120].replace("\n", "\\n")
            print(f"  [{i}] {kind} len={len(s.text)} preview={preview!r}...")

        if args.no_compress:
            print("no_compress: True (output unchanged)")
            out = raw
        else:
            out = compress_conversation_value(raw, default_compress)
            print("same_as_input:", out == raw)

        # 只打印关键信息，避免刷屏
        fixed = "".join(s.text for s in segs if not s.compressible)
        print("fixed_prefix_unchanged:", out[: len(fixed)] == fixed)
        print("raw_len:", len(raw), "out_len:", len(out))
        return

    demo = (
        "### 相关媒资提取，输出格式：文档序号#名称|类型|相关系数 ###\n"
        "示例问题？[SEP]"
        "0.短文档A\t类型：电影\n"
        "1.非常长的文档B..." + "x" * 200 + "\t类型：电影"
    )
    segs = parse_conversation_value(demo)
    assert len(segs) == 5  # header, query, SEP, doc0, doc1
    print("demo segments:", [(s.compressible, len(s.text)) for s in segs])


if __name__ == "__main__":
    main()

