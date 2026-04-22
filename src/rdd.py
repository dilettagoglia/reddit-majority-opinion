from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


THRESHOLD = 18.0
BANDWIDTH = 12.0
MIN_OBS = 50
ALL_CUTOFFS = [THRESHOLD]
OUTCOME_COLUMNS = [
    "ESH_perc",
    "NAH_perc",
    "NTA_perc",
    "YTA_perc",
    "unsure_perc",
    "no_vote_perc",
]
PERCENT_TO_COUNT_COLUMNS = {
    "ESH_perc": "ESH_count",
    "NAH_perc": "NAH_count",
    "NTA_perc": "NTA_count",
    "YTA_perc": "YTA_count",
    "unsure_perc": "unsure_count",
    "no_vote_perc": "no_vote_count",
}
OUTCOME_ORDER = [
    "no_vote_count",
    "unsure_count",
    "YTA_count",
    "ESH_count",
    "NTA_count",
    "NAH_count",
]
OUTCOME_LABELS = {
    "no_vote_count": "no_vote",
    "unsure_count": "unsure",
    "YTA_count": "YTA",
    "ESH_count": "ESH",
    "NTA_count": "NTA",
    "NAH_count": "NAH",
}
FINAL_JUDGMENTS = [
    "Not the A-hole",
    "Asshole",
    "Everyone Sucks",
    "No A-holes here",
]
FINAL_JUDGMENT_SLUGS = {
    "Not the A-hole": "nta",
    "Asshole": "yta",
    "Everyone Sucks": "esh",
    "No A-holes here": "nah",
}


def triangular_kernel(u: pd.Series) -> pd.Series:
    return np.clip(1 - np.abs(u), 0, None)


def get_plot_dir() -> Path:
    plot_dir = Path("./img/bayesian_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def ordered_outcomes(available_outcomes) -> list:
    return [outcome for outcome in OUTCOME_ORDER if outcome in available_outcomes]


def format_cutoff_label(cutoff: float) -> str:
    return f"{cutoff:.1f}h"


def prepare_rdd_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["created"] = pd.to_datetime(
        df["created"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["submission_id", "created"]).copy()

    start_time = df.groupby("submission_id")["created"].transform("min")
    df["hours_from_start"] = (df["created"] - start_time).dt.total_seconds() / 3600.0
    df = df.sort_values(["submission_id", "created"]).copy()
    df["comment_index"] = df.groupby("submission_id").cumcount() + 1

    outcomes = [col for col in OUTCOME_COLUMNS if col in df.columns]
    for outcome in outcomes:
        df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
        count_outcome = PERCENT_TO_COUNT_COLUMNS[outcome]
        df[count_outcome] = (df[outcome] * df["comment_index"]).round().astype(float)

    return df


def make_cutoff_dataset(data: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    df = data.copy()
    df["cutoff_h"] = cutoff
    df["running"] = df["hours_from_start"] - cutoff
    df["post"] = (df["running"] >= 0).astype(int)

    crosses = df.groupby("submission_id")["post"].agg(["min", "max"]).reset_index()
    valid_ids = crosses.loc[
        (crosses["min"] == 0) & (crosses["max"] == 1), "submission_id"
    ]
    return df[df["submission_id"].isin(valid_ids)].copy()


def fit_local_linear_rdd(
    data: pd.DataFrame, outcome: str, bandwidth: float = BANDWIDTH
):
    window = data.loc[
        np.abs(data["running"]) <= bandwidth, ["running", "post", outcome]
    ].dropna().copy()
    n_obs = len(window)

    if n_obs < MIN_OBS:
        return None, window

    window["weight"] = triangular_kernel(window["running"] / bandwidth)
    model = smf.wls(
        formula=f"{outcome} ~ post + running + post:running",
        data=window,
        weights=window["weight"],
    ).fit(cov_type="HC1")
    return model, window


def local_linear_rdd(data: pd.DataFrame, outcome: str, bandwidth: float = BANDWIDTH) -> dict:
    model, window = fit_local_linear_rdd(data, outcome, bandwidth=bandwidth)
    n_obs = len(window)

    if model is None:
        return {
            "cutoff_h": data["cutoff_h"].iloc[0] if "cutoff_h" in data.columns and not data.empty else np.nan,
            "outcome": outcome,
            "bandwidth_h": bandwidth,
            "n_obs": n_obs,
            "tau_at_cutoff": np.nan,
            "se_tau": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
        }

    tau = model.params.get("post", np.nan)
    se_tau = model.bse.get("post", np.nan)
    ci_low, ci_high = model.conf_int().loc["post"]

    return {
        "cutoff_h": float(data["cutoff_h"].iloc[0]) if "cutoff_h" in data.columns and not data.empty else np.nan,
        "outcome": outcome,
        "bandwidth_h": bandwidth,
        "n_obs": n_obs,
        "tau_at_cutoff": float(tau),
        "se_tau": float(se_tau),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": float(model.pvalues.get("post", np.nan)),
    }


def plot_binned_rdd_by_judgment(
    judgment_cutoff_data: dict,
    outcome: str,
    bandwidth: float = BANDWIDTH,
) -> None:
    fig, axes = plt.subplots(1, len(FINAL_JUDGMENTS), figsize=(5 * len(FINAL_JUDGMENTS), 4.5), sharey=False)
    if len(FINAL_JUDGMENTS) == 1:
        axes = [axes]

    for ax, judgment in zip(axes, FINAL_JUDGMENTS):
        data = judgment_cutoff_data.get(judgment)
        ax.set_title(judgment)

        if data is None or data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            continue

        plot_df = data.loc[np.abs(data["running"]) <= 24, ["running", "post", outcome]].dropna().copy()

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            continue

        model, window = fit_local_linear_rdd(data, outcome, bandwidth=bandwidth)
        bins = np.arange(-24, 24.5, 0.5)
        plot_df["bin"] = pd.cut(plot_df["running"], bins=bins, include_lowest=True)
        binned = (
            plot_df.groupby("bin", observed=True)
            .agg(running_mid=("running", "mean"), y_mean=(outcome, "mean"))
            .dropna()
        )

        ax.scatter(binned["running_mid"], binned["y_mean"], s=20, alpha=0.8, color="#1f4e79")

        if model is not None and not window.empty:
            left_x = np.linspace(-bandwidth, 0, 100)
            right_x = np.linspace(0, bandwidth, 100)
            left_pred = model.predict(pd.DataFrame({"running": left_x, "post": 0}))
            right_pred = model.predict(pd.DataFrame({"running": right_x, "post": 1}))
            ax.plot(left_x, left_pred, color="#c44e52", linewidth=2)
            ax.plot(right_x, right_pred, color="#c44e52", linewidth=2)

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Hours from cutoff")

    axes[0].set_ylabel(OUTCOME_LABELS.get(outcome, outcome))
    fig.suptitle(
        f"RDD binned counts around {format_cutoff_label(THRESHOLD)} cutoff: {OUTCOME_LABELS.get(outcome, outcome)}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(get_plot_dir() / f"rdd_binned_{outcome}_by_final_judgment_18.0h.png", bbox_inches="tight")
    plt.close()


def plot_rdd_effects_grid(
    results: pd.DataFrame,
    output_name: str = "rdd_effects_all_cutoffs.png",
    title: str = "RDD estimates across all cutoffs",
) -> None:
    clean = results.dropna(subset=["tau_at_cutoff", "ci95_low", "ci95_high"]).copy()
    if clean.empty:
        return

    outcomes = ordered_outcomes(clean["outcome"].unique())
    fig, axes = plt.subplots(1, len(ALL_CUTOFFS), figsize=(5 * len(ALL_CUTOFFS), 4.8), sharey=True)
    if len(ALL_CUTOFFS) == 1:
        axes = [axes]

    y_pos = np.arange(len(outcomes))
    y_labels = [OUTCOME_LABELS.get(outcome, outcome) for outcome in outcomes]

    for ax, cutoff in zip(axes, ALL_CUTOFFS):
        cutoff_df = clean.loc[clean["cutoff_h"] == cutoff].copy()
        cutoff_df["outcome"] = pd.Categorical(cutoff_df["outcome"], categories=outcomes, ordered=True)
        cutoff_df = cutoff_df.sort_values("outcome")

        lower_err = cutoff_df["tau_at_cutoff"] - cutoff_df["ci95_low"]
        upper_err = cutoff_df["ci95_high"] - cutoff_df["tau_at_cutoff"]

        ax.errorbar(
            cutoff_df["tau_at_cutoff"],
            y_pos,
            xerr=[lower_err, upper_err],
            fmt="o",
            color="#1f4e79",
            ecolor="#7a7a7a",
            capsize=4,
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"Cutoff {format_cutoff_label(cutoff)}")
        ax.set_xlabel("Estimated jump")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)

    axes[0].set_ylabel("Outcome")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(get_plot_dir() / output_name, bbox_inches="tight")
    plt.close()


def plot_rdd_effects_by_judgment(results_by_judgment: dict) -> None:
    valid_results = {
        judgment: results.dropna(subset=["tau_at_cutoff", "ci95_low", "ci95_high"]).copy()
        for judgment, results in results_by_judgment.items()
        if not results.empty
    }
    if not valid_results:
        return

    fig, axes = plt.subplots(1, len(FINAL_JUDGMENTS), figsize=(5 * len(FINAL_JUDGMENTS), 5), sharey=True)
    if len(FINAL_JUDGMENTS) == 1:
        axes = [axes]

    outcomes = ordered_outcomes(
        pd.concat(valid_results.values(), ignore_index=True)["outcome"].unique()
    )
    y_pos = np.arange(len(outcomes))
    y_labels = [OUTCOME_LABELS.get(outcome, outcome) for outcome in outcomes]

    for ax, judgment in zip(axes, FINAL_JUDGMENTS):
        judgment_df = valid_results.get(judgment)
        ax.set_title(judgment)
        if judgment_df is None or judgment_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            continue

        judgment_df["outcome"] = pd.Categorical(
            judgment_df["outcome"], categories=outcomes, ordered=True
        )
        judgment_df = judgment_df.sort_values("outcome")
        lower_err = judgment_df["tau_at_cutoff"] - judgment_df["ci95_low"]
        upper_err = judgment_df["ci95_high"] - judgment_df["tau_at_cutoff"]

        ax.errorbar(
            judgment_df["tau_at_cutoff"],
            y_pos,
            xerr=[lower_err, upper_err],
            fmt="o",
            color="#1f4e79",
            ecolor="#7a7a7a",
            capsize=4,
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Estimated jump")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)

    axes[0].set_ylabel("Outcome")
    fig.suptitle(f"RDD count estimates by final judgment at cutoff {format_cutoff_label(THRESHOLD)}", y=1.02)
    plt.tight_layout()
    plt.savefig(get_plot_dir() / "rdd_count_by_final_judgment_18.0h.png", bbox_inches="tight")
    plt.close()


def plot_running_density_grid(cutoff_data: dict) -> None:
    fig, axes = plt.subplots(1, len(ALL_CUTOFFS), figsize=(5 * len(ALL_CUTOFFS), 4.5), sharey=True)
    if len(ALL_CUTOFFS) == 1:
        axes = [axes]

    for ax, cutoff in zip(axes, ALL_CUTOFFS):
        data = cutoff_data[cutoff]
        density_df = data.loc[np.abs(data["running"]) <= 24, ["running"]].dropna().copy()
        ax.set_title(f"Cutoff {format_cutoff_label(cutoff)}")

        if density_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(
                density_df["running"],
                bins=48,
                color="#7db0a8",
                edgecolor="white",
                alpha=0.95,
            )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Hours from cutoff")

    axes[0].set_ylabel("Count")
    fig.suptitle("Running-variable density around each cutoff", y=1.02)
    plt.tight_layout()
    plt.savefig(get_plot_dir() / "rdd_running_density_all_cutoffs.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    df_base = prepare_rdd_data("./data/data-tidy/perc_disagreement_in_time.csv")
    outcomes = ordered_outcomes(df_base.columns)
    all_results = []
    cutoff_data = {}

    for cutoff in ALL_CUTOFFS:
        df_rdd = make_cutoff_dataset(df_base, cutoff)
        cutoff_data[cutoff] = df_rdd
        rdd_results = pd.DataFrame(
            [local_linear_rdd(df_rdd, outcome, bandwidth=BANDWIDTH) for outcome in outcomes]
        )
        all_results.append(rdd_results)

        print(f"\nRDD estimates at cutoff {cutoff}h:")
        print(rdd_results)

        output_csv = Path(f"./data/data-analysis/rdd_summary_{cutoff}h.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rdd_results.to_csv(output_csv, index=False)

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results["is_main_cutoff"] = combined_results["cutoff_h"].eq(THRESHOLD)
    combined_output_csv = Path("./data/data-analysis/rdd_summary_all_cutoffs.csv")
    combined_results.to_csv(combined_output_csv, index=False)

    plot_running_density_grid(cutoff_data)
    plot_rdd_effects_grid(combined_results)

    judgment_results_map = {}
    judgment_cutoff_data = {}
    for final_judgment in FINAL_JUDGMENTS:
        judgment_df = df_base.loc[df_base["final_judg"] == final_judgment].copy()
        if judgment_df.empty:
            continue

        judgment_results = []
        for cutoff in ALL_CUTOFFS:
            df_rdd = make_cutoff_dataset(judgment_df, cutoff)
            judgment_cutoff_data[final_judgment] = df_rdd
            rdd_results = pd.DataFrame(
                [local_linear_rdd(df_rdd, outcome, bandwidth=BANDWIDTH) for outcome in outcomes]
            )
            judgment_results.append(rdd_results)

        judgment_results_map[final_judgment] = pd.concat(judgment_results, ignore_index=True)

    plot_rdd_effects_by_judgment(judgment_results_map)
    for outcome in outcomes:
        plot_binned_rdd_by_judgment(judgment_cutoff_data, outcome, bandwidth=BANDWIDTH)


if __name__ == "__main__":
    main()
