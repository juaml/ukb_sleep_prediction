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

# List of the best models, taken from the results of 8-create_results_table.py
with open(results_path / "best_models.json", "r") as f:
    best_models = json.load(f)


# %% Config paths

results_path = Path(__file__).parent / "results"


# %% Load data
results_validation = pd.read_csv(
    results_path / "results_validation.csv", index_col=0, sep=";"
)

best_models_validation = {}
for t_score, t_models in best_models.items():
    best_models_validation[t_score] = []
    for t_target, t_model in t_models.items():
        t_val_df = results_validation.query(
            f"model == '{t_model}' and target == '{t_target}'"
        )
        if len(t_val_df) == 0:
            print(f"Missing {t_target} {t_model}")
            t_val_df = pd.DataFrame(
                {
                    "target": [t_target],
                    "model": [t_model],
                    "f1": [np.nan],
                    "roc_auc": [np.nan],
                    "average_precision": [np.nan],
                    "balanced_accuracy": [np.nan],
                }
            )
            t_val_df.set_index("target", inplace=True)
        best_models_validation[t_score].append(t_val_df)
    best_models_validation[t_score] = pd.concat(
        best_models_validation[t_score]
    )

# %% Plot best models
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

# %%
split = "test"

all_chance = []
chance_models = ["dummy", "dummy_stratified"]
for chance_model in chance_models:
    for t_target in targets:
        target_path = results_path / t_target
        fname = target_path / f"{t_target}_{chance_model}_cv_scores.csv"
        print(f"Loading {fname}")
        t_df = pd.read_csv(fname, sep=";", index_col=0)
        t_df["model"] = chance_model
        t_df["target"] = t_target
        all_chance.append(t_df)
chance_df = pd.concat(all_chance)
mean_chance = chance_df.groupby(["target", "model"])[
    [f"{split}_{x}" for x in scores]
].mean()
chance_levels = {
    "f1": mean_chance[f"{split}_f1"].unstack(-1),
    "average_precision": mean_chance[f"{split}_average_precision"].unstack(
        -1
    ),
    "roc_auc": mean_chance[f"{split}_roc_auc"].unstack(-1),
    "balanced_accuracy": mean_chance[f"{split}_balanced_accuracy"].unstack(
        -1
    ),
}

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
for i_score, (t_score, t_df) in enumerate(best_models_data.items()):
    this_scores = t_df[t_df["split"] == split]
    t_ax = axes.ravel()[i_score]
    sns.swarmplot(x="target", y=t_score, data=this_scores, ax=t_ax)
    sns.boxplot(
        x="target",
        y=t_score,
        data=this_scores,
        ax=t_ax,
        color="w",
        whis=(5, 95),
    )
    if t_score in chance_levels:
        for chance_model, color in zip(chance_models, ["r", "g"]):
            chance_df = chance_levels[t_score][chance_model]
            chance_df.name = t_score
            chance_df.index.name = "target"
            chance_df = chance_df.reset_index()
            sns.boxplot(
                x="target",
                y=t_score,
                data=chance_df,
                ax=t_ax,
                color="w",
                showbox=False,
                showcaps=False,
                showmeans=False,
                medianprops=dict(linestyle="--", linewidth=2, color=color),
            )
    else:
        [_, ymax] = t_ax.get_ylim()
        t_ax.set_ylim([0.5, ymax])

    # if split == "test" and t_score in ["f1", "balanced_accuracy"]:
    if split == "test":
        t_val_df = best_models_validation[t_score][t_score]
        t_val_df.name = t_score
        t_val_df.index.name = "target"
        t_val_df = t_val_df.reset_index()
        sns.boxplot(
            x="target",
            y=t_score,
            data=t_val_df,
            ax=t_ax,
            color="w",
            showbox=False,
            showcaps=False,
            showmeans=False,
            medianprops=dict(linestyle="-", linewidth=2, color="k"),
        )

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

    if i_score == 0:
        ylims = t_ax.get_ylim()
        (line1,) = t_ax.plot(
            [-1, -1], label="Baseline (majority)", c="r", ls="--", lw=2
        )
        (line2,) = t_ax.plot(
            [-1, -1], label="Baseline (chance)", c="g", ls="--", lw=2
        )
        all_lines = [line1, line2]
        if split == "test":
            (line3,) = t_ax.plot(
                [-1, -1], label="Validation", c="k", ls="-", lw=2
            )
            all_lines.append(line3)

        t_ax.legend(handles=all_lines)
        t_ax.set_ylim(ylims)
fig.subplots_adjust(hspace=0.4, top=0.95)
fig.suptitle(f"Best model's performance by metric and target (CV-{split})")
fig.savefig(f"./figs/best_models_all_{split}_zoom.pdf", bbox_inches="tight")
