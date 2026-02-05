import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_oracle(df_full, strat="mu", scale=False, save_name=None, n_reps=100):
    # _, ax = plt.subplots(figsize=(3.25, 1.75))
    # _, ax = plt.subplots(figsize=(3.6, 2.5))
    mus = df_full[strat].unique()
    print(mus)
    fig, axes = plt.subplots(1, 2, figsize=(4.25, 2.25))

    for ax, mu in zip(axes, mus):
        df = df_full[df_full[strat] == mu]
        ax.grid(True, alpha=0.3)

        # --- Oracle curve ---
        # alpha arbitrary
        oracle_df = df[(df["method"] == "oracle") & (df["alpha"] == 0.05)].sort_values(
            "interval_width"
        )

        rep_groups = [oracle_df[oracle_df["rep"] == rp] for rp in range(n_reps)]
        num_null = np.sum([len(rg) == 0 for rg in rep_groups])
        
        oracle_coverages = [
            np.mean(
                [
                    np.mean(np.abs(rg["interval_width"]) <= w) if len(rg) > 0 else 1
                    for rg in rep_groups
                ]
            )
            for w in oracle_df["interval_width"]
        ]

        # If widening is scaled by SE, scale oracle CIs too post sorting
        # if scale:
        #     # oracle_df['interval_width'] = oracle_df['interval_width'] * oracle_df['se']
        #     oracle_df['interval_width'] = oracle_df['interval_width'] * np.mean(df['se'])
        if scale:
            avg_se = np.sum(
                [
                    np.mean(rg["se"]) if len(rg) > 0 else 0
                    for rg in rep_groups
                ]
            ) / n_reps
            oracle_df["interval_width"] *= avg_se

        ax.plot(
            oracle_coverages,
            oracle_df["interval_width"],
            color="black",
            linestyle="solid",
            linewidth=2,
            label="Oracle I&W",
        )

        # --- Reference line ---
        ax.axvline(0.95, color="grey", linestyle="dotted")

        # --- Non-oracle summaries (mean over rep) ---
        summary = (
            df[df["method"] != "oracle"]
            .groupby(["method", "alpha", "rep"], as_index=False)
            .agg({"coverage": "mean", "interval_width": "mean"})
            .groupby(["method", "alpha"], as_index=False)
            .agg({"coverage": "sum", "interval_width": "mean"})
        )
        # correct error rate over models with no model selected
        summary["coverage"] = (summary["coverage"] + num_null) / n_reps

        # Split-and-condition
        df_df = summary[(summary["method"] == "hybrid")]
        ax.plot(
            # df_df["coverage"][::-1],
            1 - df_df["alpha"][::-1],
            df_df["interval_width"][::-1],
            color="forestgreen",
            linestyle="dashed",
            linewidth=2,
            label="Hybrid",
        )

        df_df = summary[(summary["method"] == "cond")]
        ax.plot(
            # df_df["coverage"][::-1],
            1 - df_df["alpha"][::-1],
            df_df["interval_width"][::-1],
            color="#984ea3",
            linestyle="dashed",
            linewidth=2,
            label="Conditional",
        )

        if "zoom_stepdown" in summary["method"].values:
            df_df = summary[(summary["method"] == "zoom_stepdown")]
            ax.plot(
                # df_df["coverage"][::-1],
                1 - df_df["alpha"][::-1],
                df_df["interval_width"][::-1],
                color="#ff7f00",
                linestyle="dashed",
                linewidth=2,
                label="Zoom",
            )

        # Classic (points)
        naive_df = summary[(summary["method"] == "naive") & (summary["alpha"] == 0.05)]
        ax.scatter(
            naive_df["coverage"],
            naive_df["interval_width"],
            color="red",
            marker="^",
            s=60,
            label="Classic (95%)",
        )

        # Infer-and-widen (points)
        as_df = summary[(summary["method"] == "LSI") & (summary["alpha"] == 0.05)]
        ax.scatter(
            as_df["coverage"],
            as_df["interval_width"],
            color="blue",
            marker="s",
            s=60,
            label="LSI (95%)",
        )

        as_df = summary[(summary["method"] == "SI") & (summary["alpha"] == 0.05)]
        ax.scatter(
            as_df["coverage"],
            as_df["interval_width"],
            color="#a65628",
            marker="s",
            s=60,
            label="SI (95%)",
        )

        # --- Axes ---s
        # ax.set_ylabel("CI Width")
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.tick_params(labelsize=11)

    axes[0].set_ylabel("CI Width")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        frameon=False,
    )

    # axes[1].legend(loc="upper center", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False)

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(
            save_name,
            # f"figures/vignette_1/vignette-1_oracle-curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def plot_oracle_randomized(df_full, save_fig=True):
    eps = sorted(df_full["eps"].unique(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(4.25, 2.25))

    for ax, ep in zip(axes, eps):
        df = df_full[df_full["eps"] == ep]
        ax.grid(True, alpha=0.3)

        # --- Oracle curve ---
        # alpha arbitrary
        oracle_df = df[(df["method"] == "oracle") & (df["alpha"] == 0.1)].sort_values(
            "interval_width"
        )

        ax.plot(
            (np.arange(len(oracle_df)) + 1) / len(oracle_df),
            oracle_df["interval_width"],
            color="black",
            linestyle="solid",
            linewidth=2,
            label="Oracle I&W",
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
        method = "fission"
        df_method = (
            summary[(summary["method"] == method)]
            .groupby("alpha", as_index=False)
            .agg({"coverage": "mean", "interval_width": "mean"})
        )
        ax.plot(
            # df_df["coverage"][::-1],
            1 - df_method["alpha"][::-1],
            df_method["interval_width"][::-1],
            color="forestgreen",
            linestyle="dashed",
            linewidth=2,
            label="Data fission",
        )

        # Classic (points)
        method = "naive"
        df_method = summary[(summary["method"] == method) & (summary["alpha"] == 0.05)]
        ax.scatter(
            df_method["coverage"],
            df_method["interval_width"],
            color="red",
            marker="^",
            s=60,
            label="Classic (95%)",
        )

        # Infer-and-widen (points)
        as_df = summary[(summary["method"] == "stability") & (summary["alpha"] == 0.05)]
        ax.scatter(
            as_df["coverage"],
            as_df["interval_width"],
            color="blue",
            marker="s",
            s=60,
            label="Algorithmic stability I&W (95%)",
        )

        # --- Axes ---s
        # ax.set_ylabel("CI Width")
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.tick_params(labelsize=11)

    axes[0].set_ylabel("CI Width")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=False,
    )

    # axes[1].legend(loc="upper center", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False)

    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f"figures/vignette_1/vignette-1_oracle-curves_randomized.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
