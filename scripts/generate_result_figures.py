from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_FIGS = ROOT / "manuscript" / "figs"
JOBHOP_RESULTS = ROOT / "output" / "jobhop_with_gru4rec_results.json"
KARRIEREWEGE_RESULTS = ROOT / "output" / "karrierewege_with_gru4rec_results.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_results(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ndcg_mean_std(obj, key):
    ndcg = obj["repeated"]["aggregated_results"][key]["NDCG@K"]
    return ndcg["mean"], ndcg["std"]


def mean_full_weights(obj):
    splits = obj["repeated"]["splits"]
    cf = np.mean([s["tuned_params"]["full_hybrid"]["cf_weight"] for s in splits])
    rl = np.mean([s["tuned_params"]["full_hybrid"]["rl_weight"] for s in splits])
    tp = np.mean([s["tuned_params"]["full_hybrid"]["topsis_weight"] for s in splits])
    return cf, rl, tp


def save_pdf(fig, target: Path):
    fallback = target.with_name(target.stem + "_build.pdf")
    try:
        fig.savefig(target, bbox_inches="tight")
    except PermissionError:
        fig.savefig(fallback, bbox_inches="tight")


def make_selected_model_plot(jobhop, karrierewege):
    models = [
        ("TOPSIS", "topsis"),
        ("RL-bandit", "rl_bandit"),
        ("GRU4Rec", "gru4rec"),
        ("SASRec", "sasrec"),
        ("Item Markov", "item_markov"),
        ("Transition-CF", "transition_cf"),
        ("CF+TOPSIS", "hybrid_cf_topsis"),
        ("CF+RL+TOPSIS", "full_hybrid"),
    ]
    datasets = [("JobHop", jobhop), ("Karrierewege", karrierewege)]
    colors = ["#b0b7c3", "#c7a26a", "#86a6d8", "#5f7db2", "#6c8ebf", "#7ea07b", "#9b8ac9", "#1f4e79"]
    all_means = []
    for _, obj in datasets:
        all_means.extend([ndcg_mean_std(obj, key)[0] for _, key in models])
    xmin = max(0.0, min(all_means) - 0.03)
    xmax = max(all_means) + 0.05

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)

    for ax, (title, obj) in zip(axes, datasets):
        means = [ndcg_mean_std(obj, key)[0] for _, key in models]
        stds = [ndcg_mean_std(obj, key)[1] for _, key in models]
        y = np.arange(len(models))
        ax.barh(y, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.6, capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, _ in models])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("NDCG@5")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        for yi, mean in enumerate(means):
            ax.text(mean + 0.004, yi, f"{mean:.3f}", va="center", ha="left", fontsize=7)

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    fig.tight_layout(w_pad=1.4)
    save_pdf(fig, MANUSCRIPT_FIGS / "fig3_selected_models.pdf")
    plt.close(fig)


def make_weight_plot(jobhop, karrierewege):
    labels = ["JobHop", "Karrierewege"]
    jobhop_w = mean_full_weights(jobhop)
    kar_w = mean_full_weights(karrierewege)
    data = np.array([jobhop_w, kar_w])
    cf = data[:, 0]
    rl = data[:, 1]
    tp = data[:, 2]
    colors = {"CF": "#6c8ebf", "RL": "#c7a26a", "TOPSIS": "#7ea07b"}

    fig, ax = plt.subplots(figsize=(5.6, 2.45))
    y = np.arange(len(labels))
    bar_height = 0.56
    edge = "#333333"
    ax.barh(y, cf, height=bar_height, color=colors["CF"], edgecolor=edge, linewidth=0.5, label="CF")
    ax.barh(y, rl, left=cf, height=bar_height, color=colors["RL"], edgecolor=edge, linewidth=0.5, label="RL")
    ax.barh(y, tp, left=cf + rl, height=bar_height, color=colors["TOPSIS"], edgecolor=edge, linewidth=0.5, label="TOPSIS")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean validation-selected weight")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=1.6, columnspacing=1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, (c, r, t) in enumerate(data):
        segments = [
            (0.0, c, f"{c:.2f}"),
            (c, r, f"{r:.2f}"),
            (c + r, t, f"{t:.2f}"),
        ]
        for left, width, label in segments:
            ax.text(left + width / 2.0, yi, label, ha="center", va="center", fontsize=7.2)

    fig.tight_layout(pad=0.6)
    save_pdf(fig, MANUSCRIPT_FIGS / "fig4_fusion_weights.pdf")
    plt.close(fig)


def main():
    jobhop = load_results(JOBHOP_RESULTS)
    karrierewege = load_results(KARRIEREWEGE_RESULTS)
    make_selected_model_plot(jobhop, karrierewege)
    make_weight_plot(jobhop, karrierewege)


if __name__ == "__main__":
    main()
