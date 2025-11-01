# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %% Config visuals
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"xtick.labelsize": 10})
sns.set_palette("colorblind")


# %% Config target/models/scores

targets = [
    "pheno_extreme",
    "sleep_duration_extreme",
    "morning_extreme",
    "morning_evening_person_extreme",
    "nap_day_extreme",
    "doze_day",
    "snoring",
]


models = [
    "gslinearsvm",
    "linearsvchc",
    "logithc",
    "gsrf",
    "gset",
    "gssvm_rbf",
    "stacked_linearsvcheursiticc",
]

model_labels = {
    "gslinearsvm": "GSLinearSVM",
    "linearsvchc": "LinearSVMHC",
    "logithc": "LogitHC",
    "gsrf": "GSRF",
    "gset": "GSET",
    "gssvm_rbf": "GSSVM-RBF",
    "stacked_linearsvcheursiticc": "Stacked",
}

scores = ["roc_auc", "balanced_accuracy", "f1", "average_precision"]

scores_labels = {
    "roc_auc": "ROC-AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1-Score",
    "average_precision": "Average Precision",
}

target_labels = {
    "pheno_extreme": "Sleeplesness/Insomnia (extremes)",
    "sleep_duration_extreme": "Sleep Duration (extremes)",
    "morning_extreme": "Getting up in the morning (extremes)",
    "morning_evening_person_extreme": "Morning/Evening Chronotype (extremes)",
    "nap_day_extreme": "Nap during day (extremes)",
    "doze_day": "Daytime Dozing",
    "snoring": "Snoring",
}

target_labels_xaxis = {
    "pheno_extreme": "Insomnia",
    "sleep_duration_extreme": "Sleep duration",
    "morning_extreme": "Getting up in the morning",
    "morning_evening_person_extreme": "Morning/Evening chronotype",
    "nap_day_extreme": "Daytime nap",
    "doze_day": "Daytime sleepiness",
    "snoring": "Snoring",
}

# %% Config paths

results_path = Path(__file__).parent / "results"
confounds_results_path = Path(__file__).parent / "results_confounds"

# List of the best models, taken from the results of 8-create_results_table.py
with open(results_path / "best_models.json", "r") as f:
    best_models = json.load(f)


# %% Load best models data

best_models_data = {}

for t_score, t_models in best_models.items():
    t_dfs = []
    for t_target, t_model in t_models.items():
        target_path = results_path / t_target
        fname = target_path / f"{t_target}_{t_model}_cv_scores.csv"
        print(f"Loading {fname}")
        t_df = pd.read_csv(fname, sep=";", index_col=0)
        t_df["model"] = t_model
        t_df["target"] = t_target
        t_df["features"] = "brain"
        t_dfs.append(t_df)

        if t_model == "stacked_linearsvcheursiticc":
            t_model = "linearsvchc"
        confounds_fname = (
            confounds_results_path
            / t_target
            / f"{t_target}_{t_model}_reduced_cv_scores.csv"
        )
        if confounds_fname.exists() is True:
            t_df = pd.read_csv(confounds_fname, sep=";", index_col=0)
            t_df["model"] = t_model
            t_df["target"] = t_target
            t_df["features"] = "age and sex"
            t_dfs.append(t_df)
        else:
            print(f"File {confounds_fname} does not exist, skipping.")

        confounds_fname = (
            confounds_results_path
            / t_target
            / f"{t_target}_{t_model}_full_cv_scores.csv"
        )

        if confounds_fname.exists() is True:
            t_df = pd.read_csv(confounds_fname, sep=";", index_col=0)
            t_df["model"] = t_model
            t_df["target"] = t_target
            t_df["features"] = "all confounds"
            t_dfs.append(t_df)
        else:
            print(f"File {confounds_fname} does not exist, skipping.")
            continue

    t_results = pd.concat(t_dfs)
    common_columns = [
        "model",
        "fold",
        "repeat",
        "n_train",
        "n_test",
        "cv_mdsum",
        "target",
        "features",
    ]

    train_scores = t_results[
        common_columns
        + [x for x in t_results.columns if x.startswith("train_")]
    ]
    test_scores = t_results[
        common_columns
        + [x for x in t_results.columns if x.startswith("test_")]
    ]

    train_scores.rename(
        columns={
            x: x.replace("train_", "")
            for x in train_scores.columns
            if x.startswith("train")
        },
        inplace=True,
    )

    test_scores.rename(
        columns={
            x: x.replace("test_", "")
            for x in test_scores.columns
            if x.startswith("test")
        },
        inplace=True,
    )

    train_scores["split"] = "train"
    test_scores["split"] = "test"
    best_models_data[t_score] = pd.concat([train_scores, test_scores])


# %% Now plot side-by-side with phenotype models
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
panel_labels = ["A", "B", "C", "D"]
i_score = 0
for t_score, t_df in best_models_data.items():
    if t_score in ["roc_auc", "average_precision"]:
        this_scores = t_df[t_df["split"] == "test"]
        t_ax = axes.ravel()[i_score]
        sns.swarmplot(
            x="target",
            y=t_score,
            data=this_scores,
            ax=t_ax,
            hue="features",
            dodge=True,
            size=4,
        )
        sns.boxplot(
            x="target",
            y=t_score,
            data=this_scores,
            ax=t_ax,
            palette=["w", "w", "w"],
            whis=(5, 95),
            hue="features",
            legend=False,
            showfliers=False,
        )

        if i_score > 0:
            t_ax.legend([], frameon=False)

        t_ax.set_xlabel("Target (Model)")
        t_ax.set_ylabel(scores_labels[t_score])
        t_ax.set_xticklabels(
            [
                f"{target_labels_xaxis[x.get_text()]}\n"
                f"({model_labels[best_models[t_score][x.get_text()]]})"
                for x in t_ax.get_xticklabels()
            ],
            rotation=60,
            fontsize=12,
        )
        t_ax.annotate(
            panel_labels[i_score],
            xy=(-0.15, 0.95),
            xycoords="axes fraction",
            fontsize=32,
            fontweight="bold",
            # xytext=(5, -5),
            textcoords="offset points",
        )
        i_score += 1
fig.subplots_adjust(hspace=0.4, top=0.90)
fig.suptitle(
    f"Comparison of models trained on brain features vs confounds (CV-test)"
)
fig.savefig(
    f"./figs/best_confounds_models_all_train_main.pdf", bbox_inches="tight"
)

# %% Now plot side-by-side with phenotype models
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
for i_score, (t_score, t_df) in enumerate(best_models_data.items()):
    this_scores = t_df[t_df["split"] == "test"]
    t_ax = axes.ravel()[i_score]
    sns.swarmplot(
        x="target",
        y=t_score,
        data=this_scores,
        ax=t_ax,
        hue="features",
        dodge=True,
        size=4,
    )
    sns.boxplot(
        x="target",
        y=t_score,
        data=this_scores,
        ax=t_ax,
        palette=["w", "w", "w"],
        whis=(5, 95),
        hue="features",
        legend=False,
        showfliers=False,
    )

    if i_score > 0:
        t_ax.legend([], frameon=False)

    if i_score > 1:
        t_ax.set_xlabel("Target (Model)")
    else:
        t_ax.set_xlabel("")
    t_ax.set_ylabel(scores_labels[t_score])
    t_ax.set_xticklabels(
        [
            f"{target_labels_xaxis[x.get_text()]}\n"
            f"({model_labels[best_models[t_score][x.get_text()]]})"
            for x in t_ax.get_xticklabels()
        ],
        rotation=60,
        fontsize=12,
    )
    t_ax.annotate(
        panel_labels[i_score],
        xy=(-0.15, 0.95),
        xycoords="axes fraction",
        fontsize=32,
        fontweight="bold",
        # xytext=(5, -5),
        textcoords="offset points",
    )
fig.subplots_adjust(hspace=0.4, top=0.95)
fig.suptitle(
    f"Comparison of models trained on brain features vs age and sex (CV-test)"
)
fig.savefig(
    f"./figs/best_confounds_models_all_train_supp.pdf", bbox_inches="tight"
)
# %%
