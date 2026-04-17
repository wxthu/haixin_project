#!/usr/bin/env python3
"""
Download RULER dataset from Hugging Face and export to JSONL.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RULER dataset and convert it to JSONL."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rbiswasfc/ruler",
        help="Hugging Face dataset name (default: rbiswasfc/ruler).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional dataset config/subset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name. If omitted, exports all splits into one JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ruler_dataset.jsonl",
        help="Output JSONL path (default: ruler_dataset.jsonl).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode when loading dataset.",
    )
    return parser.parse_args()


def iter_split_records(
    split_name: str, split_dataset: Iterable[Dict[str, Any]]
) -> Iterator[Dict[str, Any]]:
    for idx, item in enumerate(split_dataset):
        record = dict(item)
        record["_meta_split"] = split_name
        record["_meta_index_in_split"] = idx
        yield record


def get_dataset_splits(
    ds_obj: Any, chosen_split: Optional[str]
) -> Iterator[Tuple[str, Any]]:
    if hasattr(ds_obj, "keys"):
        if chosen_split is not None:
            if chosen_split not in ds_obj:
                available = ", ".join(ds_obj.keys())
                raise ValueError(
                    f"Split '{chosen_split}' not found. Available splits: {available}"
                )
            yield chosen_split, ds_obj[chosen_split]
            return

        for split_name in ds_obj.keys():
            yield split_name, ds_obj[split_name]
        return

    split_name = chosen_split or "default"
    yield split_name, ds_obj


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: 'datasets'. Install it first:\n"
            "  pip install -U datasets"
        ) from exc

    config_name = args.config
    # rbiswasfc/ruler 需要显式 config；未传时给一个默认值，避免直接报错
    if config_name is None and args.dataset == "rbiswasfc/ruler":
        config_name = "cwe_4k"
        print(
            "No --config provided for rbiswasfc/ruler, "
            "fallback to default config: cwe_4k"
        )

    print(
        f"Loading dataset='{args.dataset}', config='{config_name}', "
        f"split='{args.split}', streaming={args.streaming}"
    )

    if args.split is not None:
        ds_obj = load_dataset(
            path=args.dataset,
            name=config_name,
            split=args.split,
            streaming=args.streaming,
        )
    else:
        ds_obj = load_dataset(
            path=args.dataset,
            name=config_name,
            streaming=args.streaming,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for split_name, split_dataset in get_dataset_splits(ds_obj, args.split):
            print(f"Exporting split: {split_name}")
            for record in iter_split_records(split_name, split_dataset):
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_total += 1

    print(f"Done. Wrote {n_total} records to: {output_path}")


if __name__ == "__main__":
    main()
