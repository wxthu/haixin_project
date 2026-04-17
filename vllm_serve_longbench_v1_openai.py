import argparse
import json
import os
import time
import zipfile
from typing import Dict, List

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from openai import OpenAI
from tqdm import tqdm


# LongBench v1 official task list.
LONG_BENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

# LongBench-E task list from official README.
LONG_BENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LongBench v1 with vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--base-url", type=str, default="http://localhost:8090/v1")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument(
        "--dataset-repo", type=str, default="THUDM/LongBench", help="HF dataset repo."
    )
    parser.add_argument(
        "--use-e",
        action="store_true",
        help="Evaluate on LongBench-E split variants (dataset_e).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Empty means official full list.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=">0 for debug run with first N samples per dataset.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only download/cache datasets and print stats, no inference.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="pred_openai",
        help="Root folder for prediction outputs.",
    )
    return parser.parse_args()


def default_datasets(use_e: bool) -> List[str]:
    return LONG_BENCH_E_DATASETS if use_e else LONG_BENCH_DATASETS


def resolve_datasets(args: argparse.Namespace) -> List[str]:
    if not args.datasets.strip():
        return default_datasets(args.use_e)
    items = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not items:
        raise ValueError("No valid dataset names found in --datasets.")
    return items


def build_messages(sample: Dict) -> List[Dict[str, str]]:
    context = str(sample.get("context", "")).strip()
    question = str(sample.get("input", "")).strip()
    all_classes = sample.get("all_classes", None)
    language = str(sample.get("language", "")).lower()

    if language == "zh" or "_zh" in str(sample.get("dataset", "")):
        system_prompt = (
            "你是一个长上下文阅读与问答助手。"
            "请只基于给定上下文作答，不要编造。"
            "如果是分类任务，答案必须从给定候选类别中选择。"
            "输出简洁答案，不要添加多余解释。"
        )
    else:
        system_prompt = (
            "You are a long-context QA assistant."
            " Answer strictly based on the provided context."
            " If this is a classification task, choose from provided classes only."
            " Return a concise final answer without extra explanation."
        )

    class_hint = ""
    if isinstance(all_classes, list) and all_classes:
        class_hint = f"\nCandidate classes: {all_classes}\n"

    user_prompt = (
        "Context:\n"
        "----- BEGIN CONTEXT -----\n"
        f"{context}\n"
        "------ END CONTEXT ------\n\n"
        f"Question/Input:\n{question}\n"
        f"{class_hint}\n"
        "Provide the final answer only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_task_dataset_from_zip(repo: str, config_name: str) -> List[Dict]:
    zip_candidates = ["data.zip", "LongBench/data.zip"]
    zip_path = None
    last_err = None
    for filename in zip_candidates:
        try:
            zip_path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
            break
        except Exception as e:
            last_err = e
            continue
    if zip_path is None:
        raise RuntimeError(
            f"Failed to download LongBench data.zip from repo '{repo}'. Last error: {last_err}"
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target_name = None
        for name in names:
            if name.endswith(f"/{config_name}.jsonl") or name == f"{config_name}.jsonl":
                target_name = name
                break
        if target_name is None:
            raise RuntimeError(
                f"Cannot find '{config_name}.jsonl' inside {zip_path}. "
                f"Example files: {names[:8]}"
            )

        records: List[Dict] = []
        with zf.open(target_name, "r") as f:
            for raw in f:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def load_task_dataset(repo: str, dataset_name: str, use_e: bool):
    config_name = f"{dataset_name}_e" if use_e else dataset_name
    try:
        return load_dataset(repo, config_name, split="test")
    except Exception as e:
        # datasets>=3 may reject old script-based loaders (LongBench.py).
        msg = str(e)
        if "Dataset scripts are no longer supported" not in msg and "LongBench.py" not in msg:
            raise

        fallback_repo = repo
        if repo.lower() == "thudm/longbench":
            # Mirror repo often has data.zip and avoids script loader path.
            fallback_repo = "zai-org/LongBench"
        print(
            f"[warn] load_dataset failed for repo={repo}, config={config_name}. "
            f"Fallback to zip loader from {fallback_repo}..."
        )
        return _load_task_dataset_from_zip(fallback_repo, config_name)


def _limit_samples(ds, max_samples: int):
    if max_samples <= 0:
        return ds
    if hasattr(ds, "select"):
        return ds.select(range(min(max_samples, len(ds))))
    return ds[:max_samples]


def _first_item(ds):
    if len(ds) == 0:
        return {}
    return ds[0]


def evaluate_dataset(
    client: OpenAI,
    args: argparse.Namespace,
    dataset_name: str,
    out_path: str,
) -> None:
    ds = load_task_dataset(args.dataset_repo, dataset_name, args.use_e)
    ds = _limit_samples(ds, args.max_samples)

    total = len(ds)
    print(f"[{dataset_name}] samples={total}, output={out_path}")

    with open(out_path, "w", encoding="utf-8") as f_out:
        for sample in tqdm(ds, total=total, desc=f"{dataset_name}"):
            messages = build_messages(sample)
            start = time.perf_counter()

            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=False,
            )
            pred = response.choices[0].message.content or ""
            e2e = time.perf_counter() - start

            record = {
                # Official LongBench eval fields:
                "pred": pred.strip(),
                "answers": sample.get("answers", []),
                "all_classes": sample.get("all_classes", None),
                "length": sample.get("length", None),
                # Extra trace fields for debugging:
                "_id": sample.get("_id", None),
                "dataset": sample.get("dataset", dataset_name),
                "latency_seconds": round(e2e, 4),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_only(args: argparse.Namespace, datasets: List[str]) -> None:
    for name in datasets:
        ds = load_task_dataset(args.dataset_repo, name, args.use_e)
        n = len(ds)
        first = _first_item(ds)
        print(
            f"[prepare] {name}: samples={n}, "
            f"fields={list(first.keys()) if first else []}"
        )


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args)

    if args.prepare_only:
        prepare_only(args, datasets)
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model_tag = args.model_name.replace("/", "__")
    out_dir = os.path.join(
        args.output_root, "pred_e" if args.use_e else "pred", model_tag
    )
    ensure_dir(out_dir)

    for dataset_name in datasets:
        out_path = os.path.join(out_dir, f"{dataset_name}.jsonl")
        evaluate_dataset(client, args, dataset_name, out_path)

    print(f"Done. Prediction files are saved under: {out_dir}")


if __name__ == "__main__":
    main()

