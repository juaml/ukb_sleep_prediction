# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "gset",
    "gsrf",
    "gssvm_rbf",
    "gslinearsvm",
    "linearsvchc",
    "logithc",
    "stacked_linearsvcheursiticc",
]

model_labels = {
    "gslinearsvm": "Linear SVM",
    "linearsvchc": "Linear SVM (Heuristic C)",
    "logithc": "Logit (Heuristic C)",
    "gsrf": "Random Forest",
    "gset": "Extra Trees",
    "gssvm_rbf": "SVM (RBF Kernel)",
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
    "morning_extreme": "Easiness Getting up in the morning (extremes)",
    "morning_evening_person_extreme": "Morning/Evening Chronotype (extremes)",
    "nap_day_extreme": "Nap during day (extremes)",
    "doze_day": "Daytime Dozing",
    "snoring": "Snoring",
}

target_labels_xaxis = {
    "pheno_extreme": "Sleeplesness/Insomnia",
    "sleep_duration_extreme": "Sleep Duration",
    "morning_extreme": "Getting up",
    "morning_evening_person_extreme": "Morning/Evening",
    "nap_day_extreme": "Nap during day",
    "doze_day": "Daytime Dozing",
    "snoring": "Snoring",
}


results_path = Path(__file__).parent / "results"


# %% Load data

all_dfs = []

for t_model in models:
    for t_target in targets:
        target_path = results_path / t_target
        fname = target_path / f"{t_target}_{t_model}_cv_scores.csv"
        print(f"Loading {fname}")
        t_df = pd.read_csv(fname, sep=";", index_col=0)
        t_df["model"] = t_model
        t_df["target"] = t_target

        all_dfs.append(t_df)
t_results = pd.concat(all_dfs)

# %%
common_columns = [
    "model",
    "fold",
    "repeat",
    "n_train",
    "n_test",
    "cv_mdsum",
    "target",
]

train_scores = t_results[
    common_columns + [x for x in t_results.columns if x.startswith("train_")]
]
test_scores = t_results[
    common_columns + [x for x in t_results.columns if x.startswith("test_")]
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
        x: x.replace("test_", "") for x in test_scores.columns if x.startswith("test")
    },
    inplace=True,
)

train_scores["split"] = "train"
test_scores["split"] = "test"
final_results = pd.concat([train_scores, test_scores])

# %%
summary = final_results.groupby(["split", "target", "model", "repeat"])[scores].mean()
summary = summary.groupby(["split", "target", "model"]).mean()
summary = summary[["roc_auc", "balanced_accuracy", "f1", "average_precision"]]

# %% Find best model per target/score pair
best_models = {}
for t_score in scores:
    best_models[t_score] = {}
    print("===================================")
    print(f"Best models for {t_score}")
    t_max = summary.iloc[
        summary.reset_index()
        .query("split == 'test'")
        .groupby("target")[t_score]
        .idxmax()
    ][t_score]
    print(t_max)
    for row in t_max.items():
        best_models[t_score][row[0][1]] = row[0][2]
    print("===================================\n")

with open(results_path / "best_models.json", "w") as f:
    json.dump(best_models, f, indent=4)

# %%
summary_std = final_results.groupby(["split", "target", "model", "repeat"])[
    scores
].std()
summary_std = summary_std.groupby(["split", "target", "model"]).mean()
summary_std = summary_std[["roc_auc", "balanced_accuracy", "f1", "average_precision"]]

# %%
for t_col in summary.columns:
    summary[t_col] = (
        summary[t_col].round(3).astype(str)
        + " ± "
        + summary_std[t_col].round(3).astype(str)
    )
# %%
summary.index.names = ["Split", "Target", "Model"]
summary = summary.reset_index()
summary.replace(
    {"Target": target_labels_xaxis, "Model": model_labels},
    inplace=True,
)
summary.rename(columns=scores_labels, inplace=True)

summary.to_csv(results_path / "results_table.csv", index=False, sep=";")
# %%
