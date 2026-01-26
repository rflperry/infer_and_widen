import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_oracle(df, label, conditional=False, save_fig=True):
    # _, ax = plt.subplots(figsize=(3.25, 1.75))
    _, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.grid(True, alpha=0.3)

    # --- Oracle curve ---
    oracle_df = (
        df[(df["method"] == "oracle") & (df['alpha'] == 0.9)]
        .sort_values("interval_width")
    )

    ax.plot(
        (np.arange(len(oracle_df)) + 1) / len(oracle_df),
        oracle_df["interval_width"],
        color="black",
        linestyle="solid",
        linewidth=2,
        label="Oracle I&W"
    )

    # --- Reference line ---
    ax.axvline(0.95, color="grey", linestyle="dotted")

    # --- Non-oracle summaries (mean over rep) ---
    summary = (
        df[df["method"] != "oracle"]
        .groupby(["method", "alpha"], as_index=False)
        .agg({"coverage": "mean", "interval_width": "mean"})
    )

    # Split-and-condition
    df_df = summary[(summary["method"] == "hybrid")].groupby("alpha", as_index=False).agg({"coverage": "mean", "interval_width": "mean"})
    ax.plot(
        # df_df["coverage"][::-1],
        1 - df_df['alpha'][::-1],
        df_df["interval_width"][::-1],
        color="forestgreen", linestyle="dashed",
        linewidth=2,
        label="Hybrid"
    )

    if conditional:
        df_df = summary[(summary["method"] == "cond")].groupby("alpha", as_index=False).agg({"coverage": "mean", "interval_width": "median"})
        ax.plot(
            # df_df["coverage"][::-1],
        1 - df_df['alpha'][::-1],
            df_df["interval_width"][::-1],
            color="#984ea3", linestyle="dashed",
            linewidth=2,
            label="Conditional"
        )

    df_df = summary[(summary["method"] == "zoom_stepdown")].groupby("alpha", as_index=False).agg({"coverage": "mean", "interval_width": "mean"})
    ax.plot(
        # df_df["coverage"][::-1],
        1 - df_df['alpha'][::-1],
        df_df["interval_width"][::-1],
        color="#ff7f00", linestyle="dashed",
        linewidth=2,
        label="Zoom"
    )

    # Classic (points)
    naive_df = summary[(summary["method"] == "naive") & (summary['alpha'] == 0.05)]
    ax.scatter(
        naive_df["coverage"],
        naive_df["interval_width"],
        color="red", marker="^", s=60,
        label="Classic (95%)"
    )

    # Infer-and-widen (points)
    as_df = summary[(summary["method"] == "LSI") & (summary['alpha'] == 0.05)]
    ax.scatter(
        as_df["coverage"],
        as_df["interval_width"],
        color="blue", marker="s", s=60,
        label="LSI (95%)"
    )

    as_df = summary[(summary["method"] == "SI") & (summary['alpha'] == 0.05)]
    ax.scatter(
        as_df["coverage"],
        as_df["interval_width"],
        color="#a65628", marker="s", s=60,
        label="SI (95%)"
    )

    # --- Axes ---
    ax.set_ylabel("CI Width")
    ax.set_xlabel("Coverage")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 3))

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, frameon=False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f"figures/vignette_1/vignette-1_oracle-curves_{label}.png",
            dpi=300
        )
        plt.close()
    else:
        plt.show()