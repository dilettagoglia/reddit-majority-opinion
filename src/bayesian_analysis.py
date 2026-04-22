import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from entropy_class import Entropy

warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None

MAIN_THRESHOLD = 18
PLACEBO_THRESHOLDS = [6, 12, 24, 30]
RANDOM_SEED = 8927

np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89
sns.set(style="whitegrid")

SOURCE_DF = pd.read_csv("./data/data-tidy/perc_disagreement_in_time.csv")
SOURCE_DF["text_flair"] = SOURCE_DF["text_flair"].fillna("")
VAR_NAMES = ["ESH", "NAH", "NTA", "YTA", "unsure", "no_vote"]
PERCENT_COLUMNS = [
    "ESH_perc",
    "NAH_perc",
    "NTA_perc",
    "YTA_perc",
    "unsure_perc",
    "no_vote_perc",
]


def bayesian_preprocessing(threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = Entropy()
    df_before_merged = pd.DataFrame()
    df_after_merged = pd.DataFrame()

    total_threads = SOURCE_DF.submission_id.nunique()
    pbar = tqdm(total=total_threads, desc=f"Preprocessing {threshold}h")

    for _, sub_df in SOURCE_DF.groupby("submission_id"):
        sub_df = sub_df.copy().reset_index(drop=True)
        sub_df["created"] = pd.to_datetime(
            sub_df["created"], format="%Y-%m-%d %H:%M:%S"
        )
        sub_df[PERCENT_COLUMNS] = sub_df[PERCENT_COLUMNS].astype(float)

        thread_duration = sub_df.created.max() - sub_df.created.min()
        if thread_duration < pd.Timedelta(hours=threshold):
            pbar.update(1)
            continue

        start = sub_df.iloc[0]["created"]
        cutoff_time = start + pd.Timedelta(hours=threshold)

        df_before = sub_df[sub_df["created"] < cutoff_time].copy().reset_index(drop=True)
        df_after = sub_df[sub_df["created"] > cutoff_time].copy().reset_index(drop=True)

        if df_before.empty or df_after.empty:
            pbar.update(1)
            continue

        assert df_before.created.max() < cutoff_time
        assert df_after.created.min() > cutoff_time

        df_after[PERCENT_COLUMNS] = df_after.apply(
            lambda row: pd.Series(
                e.compute_post_entropy(
                    df_after[df_after.created <= row.created], entropy_param=False
                )
            ),
            axis=1,
        )

        df_before_merged = pd.concat(
            [df_before_merged, df_before.iloc[-1:]], ignore_index=True
        )
        df_after_merged = pd.concat(
            [df_after_merged, df_after.iloc[-1:]], ignore_index=True
        )
        pbar.update(1)

    pbar.close()

    before_path = f"./data/data-tidy/before_{threshold}h.csv"
    after_path = f"./data/data-tidy/after_{threshold}h.csv"
    df_before_merged.to_csv(before_path, index=False)
    df_after_merged.to_csv(after_path, index=False)
    return df_before_merged, df_after_merged


def load_or_preprocess_threshold(threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    before_path = f"./data/data-tidy/before_{threshold}h.csv"
    after_path = f"./data/data-tidy/after_{threshold}h.csv"

    if os.path.exists(before_path) and os.path.exists(after_path):
        print(f"Data already preprocessed for {threshold}h, loading from csv...")
        df_before_merged = pd.read_csv(before_path)
        df_after_merged = pd.read_csv(after_path)
    else:
        print(f"Preprocessing data for {threshold}h...")
        df_before_merged, df_after_merged = bayesian_preprocessing(threshold)

    df_before_merged["created"] = pd.to_datetime(
        df_before_merged["created"], format="%Y-%m-%d %H:%M:%S"
    )
    df_after_merged["created"] = pd.to_datetime(
        df_after_merged["created"], format="%Y-%m-%d %H:%M:%S"
    )
    return df_before_merged, df_after_merged


def plot_prior(df_input: pd.DataFrame, title: str, output_name: str) -> None:
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))

    for subplot, column, label in zip(
        ax.flatten(),
        PERCENT_COLUMNS,
        ["ESH", "NAH", "NTA", "YTA", "unsure", "no judg"],
    ):
        sns.histplot(df_input[column], bins=50, kde=True, ax=subplot)
        subplot.set_xlabel(label)
        subplot.set_ylabel("")

    plt.suptitle(title)
    plt.savefig(output_name)
    plt.close(fig)


def prepare_model_inputs(
    df_before_merged: pd.DataFrame, df_after_merged: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[np.ndarray], list[float], np.ndarray, list[str]]:
    df = df_before_merged.merge(
        df_after_merged, on="submission_id", suffixes=("_before", "_after")
    )

    variables = [
        "ESH_perc_before",
        "ESH_perc_after",
        "NAH_perc_before",
        "NAH_perc_after",
        "NTA_perc_before",
        "NTA_perc_after",
        "YTA_perc_before",
        "YTA_perc_after",
        "unsure_perc_before",
        "unsure_perc_after",
        "no_vote_perc_before",
        "no_vote_perc_after",
    ]
    df[variables] = df[variables].astype(float)
    df.dropna(inplace=True)
    df.drop(columns=["created_before", "created_after"], inplace=True)

    le = LabelEncoder()
    le.fit(df.final_judg_before)
    df["final_judg_before"] = le.transform(df.final_judg_before) + 1
    verdicts = list(le.classes_)
    verdict_index = df["final_judg_before"].values

    model_vars = [
        [df["ESH_perc_before"].values, df["ESH_perc_after"].values],
        [df["NAH_perc_before"].values, df["NAH_perc_after"].values],
        [df["NTA_perc_before"].values, df["NTA_perc_after"].values],
        [df["YTA_perc_before"].values, df["YTA_perc_after"].values],
        [df["unsure_perc_before"].values, df["unsure_perc_after"].values],
        [df["no_vote_perc_before"].values, df["no_vote_perc_after"].values],
    ]
    means = [
        df["ESH_perc_before"].mean(),
        df["NAH_perc_before"].mean(),
        df["NTA_perc_before"].mean(),
        df["YTA_perc_before"].mean(),
        df["unsure_perc_before"].mean(),
        df["no_vote_perc_before"].mean(),
    ]

    print("Possible verdicts:", verdicts)
    print("Encoded verdicts:", verdict_index)
    return df, variables, model_vars, means, verdict_index, verdicts


def run_bayesian_models(
    model_vars: list[np.ndarray],
    means: list[float],
    verdict_index: np.ndarray,
    verdicts: list[str],
) -> list:
    posteriors = []

    for i, var_pair in enumerate(model_vars):
        with pm.Model():
            prior_mean = np.mean(var_pair[0])
            prior_sd = np.std(var_pair[0])
            alpha = pm.Normal("alpha", prior_mean, prior_sd, shape=len(verdicts))
            sigma = pm.Uniform("sigma", 0, 1)
            beta = pm.Normal("beta", 0, 10, shape=len(verdicts))

            mu = alpha[verdict_index - 1] + beta[verdict_index - 1] * (
                var_pair[0] - means[i]
            )

            pm.Normal("mu", mu, sigma, observed=var_pair[1])

            posterior = pm.sample(
                draws=4600,
                tune=4600,
                return_inferencedata=True,
                progressbar=True,
                step=[pm.NUTS(target_accept=0.95, max_treedepth=30)],
                cores=1,
                random_seed=RANDOM_SEED,
            )
            posteriors.append(posterior)
            print(az.summary(posterior, hdi_prob=0.89))

    return posteriors


def verdict_label_map(verdicts: list[str]) -> dict[str, str]:
    label_lookup = {
        "Asshole": "YTA",
        "Everyone Sucks": "ESH",
        "No A-holes here": "NAH",
        "Not the A-hole": "NTA",
    }
    return {f"[{i}]": f"[{label_lookup.get(verdict, verdict)}]" for i, verdict in enumerate(verdicts)}


def write_summary_file(
    threshold: int,
    df_before_merged: pd.DataFrame,
    df_after_merged: pd.DataFrame,
    posteriors: list,
    verdicts: list[str],
) -> None:
    summary_path = f"./data/data-analysis/bayesian_summaries_placebo_{threshold}h.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Bayesian analysis\n")
        f.write("-----------------\n")
        f.write(f"Placebo threshold: {threshold}h\n")
        f.write(f"Main threshold reference: {MAIN_THRESHOLD}h\n")
        f.write("Threads before threshold: " + str(len(df_before_merged)) + "\n")
        f.write("Threads after threshold: " + str(len(df_after_merged)) + "\n")
        f.write("\n\n")

        for name, posterior in zip(
            [
                "Judgment expressed ESH",
                "Judgment expressed NAH",
                "Judgment expressed NTA",
                "Judgment expressed YTA",
                "Judgment expressed unsure",
                "NO Judgment expressed",
            ],
            posteriors,
        ):
            f.write(f"{name}:\n")
            f.write(az.summary(posterior, hdi_prob=0.89).to_string())
            f.write("\n\n")

    with open(summary_path, "r", encoding="utf-8") as f:
        data = f.read()

    for original, replacement in verdict_label_map(verdicts).items():
        data = data.replace(original, replacement)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(data)


def write_posterior_file(threshold: int, posteriors: list) -> None:
    posterior_path = f"./data/data-analysis/bayesian_posteriors_{threshold}h.txt"
    with open(posterior_path, "w", encoding="utf-8") as f:
        for posterior, name in zip(posteriors, VAR_NAMES):
            f.write(f"Posterior {name}:\n")
            f.write(posterior.to_dataframe().to_string())
            f.write("\n\n")


def plot_posteriors(posteriors: list, threshold: int) -> None:
    for posterior, name in zip(posteriors, VAR_NAMES):
        az.plot_posterior(posterior, var_names="beta", figsize=(10, 5))
        plt.suptitle(f"{name} beta posterior ({threshold}h)", fontsize=16)
        plt.tight_layout()
        plt.show()


def run_threshold_analysis(threshold: int) -> None:
    print(f"\nRunning placebo Bayesian analysis for {threshold}h")
    df_before_merged, df_after_merged = load_or_preprocess_threshold(threshold)

    if df_before_merged.empty or df_after_merged.empty:
        raise ValueError(f"No valid threads available for threshold {threshold}h")

    thread_id_example = df_before_merged.submission_id.iloc[0]
    print("Thread id:", thread_id_example)
    print(
        "Before:",
        df_before_merged[
            df_before_merged.submission_id == thread_id_example
        ][PERCENT_COLUMNS].values,
    )
    print(
        "After:",
        df_after_merged[df_after_merged.submission_id == thread_id_example][
            PERCENT_COLUMNS
        ].values,
    )

    plot_prior(
        df_before_merged,
        title=f"Before {threshold}h",
        output_name=f"./img/bayesian_plots/priors_before_{threshold}h.png",
    )
    plot_prior(
        df_after_merged,
        title=f"After {threshold}h",
        output_name=f"./img/bayesian_plots/priors_after_{threshold}h.png",
    )

    _, _, model_vars, means, verdict_index, verdicts = prepare_model_inputs(
        df_before_merged, df_after_merged
    )
    posteriors = run_bayesian_models(model_vars, means, verdict_index, verdicts)
    write_summary_file(
        threshold, df_before_merged, df_after_merged, posteriors, verdicts
    )
    write_posterior_file(threshold, posteriors)
    plot_posteriors(posteriors, threshold)


def main() -> None:
    for threshold in PLACEBO_THRESHOLDS:
        run_threshold_analysis(threshold)


if __name__ == "__main__":
    main()
