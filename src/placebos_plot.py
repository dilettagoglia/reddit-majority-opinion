from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path("./data/data-analysis/bayesian_summaries_all_with_placebos.csv")
OUTPUT_DIR = Path("./img/bayesian_plots")
OUTPUT_PATH = OUTPUT_DIR / "bayesian_placebo_betas.png"

VERDICT_ORDER = ["YTA", "ESH", "NTA", "NAH"]
MAIN_THRESHOLD = 18
PLACEBO_THRESHOLDS = [6, 12]

STYLE_MAP = {
    "YTA": {"color": "#c0392b", "linestyle": "-", "label": "YTA"},
    "ESH": {"color": "#c0392b", "linestyle": ":", "label": "ESH"},
    "NTA": {"color": "#1f77b4", "linestyle": "-", "label": "NTA"},
    "NAH": {"color": "#1f77b4", "linestyle": ":", "label": "NAH"},
}


def load_bayesian_summaries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1)
    df = df[df["timestamp_hours"] <=18]
    df = df.dropna(subset=["timestamp_hours", "judgment_expressed", "coeff_verdict"]).copy()
    df["timestamp_hours"] = pd.to_numeric(df["timestamp_hours"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["hdi_5.5%"] = pd.to_numeric(df["hdi_5.5%"], errors="coerce")
    df["hdi_94.5%"] = pd.to_numeric(df["hdi_94.5%"], errors="coerce")
    df = df.dropna(subset=["timestamp_hours", "mean", "hdi_5.5%", "hdi_94.5%"]).copy()

    df["timestamp_hours"] = df["timestamp_hours"].astype(int)
    df["coeff_verdict"] = df["coeff_verdict"].str.extract(r"beta\[(.+)\]")
    df = df[df["judgment_expressed"].isin(VERDICT_ORDER)].copy()
    df = df[df["coeff_verdict"].isin(VERDICT_ORDER)].copy()
    return df


def get_plot_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def add_cutoff_lines(ax: plt.Axes) -> None:
    for cutoff in PLACEBO_THRESHOLDS:
        ax.axvline(cutoff, color="#bdbdbd", linestyle="--", linewidth=1.0, zorder=0)
    ax.axvline(MAIN_THRESHOLD, color="#bdbdbd", linestyle="--", linewidth=1.0, zorder=0)
    ax.axhline(0, color="#d9d9d9", linestyle="-", linewidth=0.9, zorder=0)


def plot_placebo_betas(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharex=True, sharey=True)
    axes = list(axes)

    legend_handles = []
    legend_labels = []

    for ax, expressed_judgment in zip(axes, VERDICT_ORDER):
        panel_df = df[df["judgment_expressed"] == expressed_judgment].copy()
        add_cutoff_lines(ax)

        for coeff_verdict in VERDICT_ORDER:
            verdict_df = panel_df[panel_df["coeff_verdict"] == coeff_verdict].sort_values(
                "timestamp_hours"
            )
            if verdict_df.empty:
                continue

            style = STYLE_MAP[coeff_verdict]
            lower_err = verdict_df["mean"] - verdict_df["hdi_5.5%"]
            upper_err = verdict_df["hdi_94.5%"] - verdict_df["mean"]

            container = ax.errorbar(
                verdict_df["timestamp_hours"],
                verdict_df["mean"],
                yerr=[lower_err, upper_err],
                fmt="o",
                color=style["color"],
                ecolor=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label=style["label"],
                zorder=3,
            )

            if expressed_judgment == VERDICT_ORDER[0]:
                legend_handles.append(container.lines[0])
                legend_labels.append(style["label"])

        ax.set_title(f"Expressed judgment: {expressed_judgment}")
        ax.set_xticks(PLACEBO_THRESHOLDS[:2] + [MAIN_THRESHOLD] + PLACEBO_THRESHOLDS[2:])
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Beta coeff (verdict impact)")

    fig.suptitle("Bayesian beta estimates across placebo and main thresholds", y=0.98)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        title="Verdicts",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_bayesian_summaries(INPUT_CSV)
    get_plot_dir()
    plot_placebo_betas(df, OUTPUT_PATH)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
