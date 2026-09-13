"""Internal diagnostic figures. Not manuscript assets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .sweep import SWEEP_ROOT, assert_sweep_write


def _read(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            out = dict(r)
            for k, v in r.items():
                if k in ("seed", "source", "operator", "subject"):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            if "seed" in out:
                try:
                    out["seed"] = int(out["seed"])
                except ValueError:
                    pass
            if "k" in out:
                out["k"] = int(float(out["k"]))
            rows.append(out)
    return rows


def _mean_by(rows, keys, y):
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r[y])
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


SRC_COLOR = {"M1": "#1f4e79", "M2": "#c45c26", "M3": "#2d6a4f"}
OP_LS = {"late_product": "-", "late_mean": "--", "cpsa": ":"}
OP_LABEL = {"late_product": "late-product", "late_mean": "late-mean", "cpsa": "CPSA"}


def _save(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    pdf = assert_sweep_write(SWEEP_ROOT / "figures" / f"{name}.pdf")
    png = assert_sweep_write(SWEEP_ROOT / "figures" / f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_grid(rows, y, ylabel, name):
    import matplotlib.pyplot as plt

    ops = ("late_product", "late_mean", "cpsa")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6), sharey=False)
    for ax, op in zip(axes, ops):
        for src in ("M1", "M2", "M3"):
            sub = [r for r in rows if r["operator"] == op and r["source"] == src]
            by_k = defaultdict(list)
            for r in sub:
                by_k[r["k"]].append(r[y])
            ks = sorted(by_k)
            ys = [float(np.mean(by_k[k])) for k in ks]
            ax.plot(ks, ys, color=SRC_COLOR[src], ls=OP_LS[op], marker="o", ms=3, lw=1.2, label=src)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 8, 16, 32, 64, 100])
        ax.set_xticklabels(["1", "2", "3", "5", "8", "16", "32", "64", "100"])
        ax.set_title(OP_LABEL[op])
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, name)


def render_figures() -> list[str]:
    _style()
    metrics = _read(SWEEP_ROOT / "multiplicity_metrics.csv")
    attn = _read(SWEEP_ROOT / "cpsa_attention_diagnostics.csv")

    _plot_metric_grid(metrics, "accuracy", "Accuracy", "fig_a_accuracy")
    _plot_metric_grid(metrics, "nll", "NLL", "fig_a_nll")
    _plot_metric_grid(metrics, "brier", "Brier", "fig_a_brier")
    _plot_metric_grid(metrics, "ece", "ECE-15", "fig_a_ece")
    _plot_metric_grid(metrics, "mean_l1_to_k1", r"mean $L_1(F_k,F_1)$", "fig_b_l1_to_k1")
    _plot_metric_grid(metrics, "prediction_flip_rate", "Prediction-flip rate", "fig_b_flip")
    _plot_metric_grid(metrics, "mean_l1_to_pj", r"mean $L_1(F_k,p_j)$", "fig_b_l1_to_pj")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for src in ("M1", "M2", "M3"):
        sub = [r for r in attn if r["source"] == src]
        by_k = defaultdict(list)
        by_k_l1 = defaultdict(list)
        for r in sub:
            by_k[r["k"]].append(r["attn_mean"])
            by_k_l1[r["k"]].append(r["mean_l1_to_pj"])
        ks = sorted(by_k)
        axes[0].plot(ks, [float(np.mean(by_k[k])) for k in ks], color=SRC_COLOR[src], marker="o", ms=3, lw=1.2, label=src)
        axes[1].plot(ks, [float(np.mean(by_k_l1[k])) for k in ks], color=SRC_COLOR[src], marker="o", ms=3, lw=1.2, label=src)
    axes[0].set_title("CPSA duplicate-group attention")
    axes[0].set_ylabel(r"mean $A_j(k)$")
    axes[1].set_title("CPSA attraction to $p_j$")
    axes[1].set_ylabel(r"mean $L_1(F_k,p_j)$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 8, 16, 32, 64, 100])
        ax.set_xticklabels(["1", "2", "3", "5", "8", "16", "32", "64", "100"])
        ax.set_xlabel("k")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_c_cpsa")
    return [
        "fig_a_accuracy",
        "fig_a_nll",
        "fig_a_brier",
        "fig_a_ece",
        "fig_b_l1_to_k1",
        "fig_b_flip",
        "fig_b_l1_to_pj",
        "fig_c_cpsa",
    ]
