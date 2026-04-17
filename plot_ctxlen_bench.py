import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = [r for r in data["results"] if r.get("type") == "ok"]
    return data["meta"], results


def group_by_batch_size(results):
    by_bs = {}
    for r in results:
        bs = r["batch_size"]
        by_bs.setdefault(bs, []).append(r)
    for bs in by_bs:
        by_bs[bs].sort(key=lambda x: x["context_len"])
    return by_bs


def plot_file(path: str, ax):
    meta, results = load_results(path)
    by_bs = group_by_batch_size(results)

    for bs, items in sorted(by_bs.items()):
        ctx_lens = [it["context_len"] for it in items]
        tokps = [it["decode_new_tok_per_s_avg"] for it in items]
        ax.plot(ctx_lens, tokps, marker="o", label=f"batch_size={bs}")

    title = f"{meta.get('impl', '')} - {meta.get('model', '')}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("context_len (tokens)")
    ax.set_ylabel("decode_new_tok_per_s_avg")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)


def main():
    seer_path = "seer_decode_ctxlen_bench.json"
    hf_path = "hf_ctxlen_bench.json"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_file(seer_path, axes[0])
    plot_file(hf_path, axes[1])

    fig.suptitle("Context length vs decode_new_tok_per_s_avg", fontsize=12)
    plt.tight_layout()
    out_path = "ctxlen_bench_decode_tokps.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

