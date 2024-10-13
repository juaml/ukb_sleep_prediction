# %%
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
)


targets = [
    "doze_day",
    "morning_evening_person_extreme",
    "morning_extreme",
    "nap_day_extreme",
    "pheno_extreme",
    "sleep_duration_extreme",
    "snoring",
]


target_labels = {
    "pheno_extreme": "Sleeplesness/Insomnia (extremes)",
    "sleep_duration_extreme": "Sleep Duration (extremes)",
    "morning_extreme": "Easiness getting up in the morning (extremes)",
    "morning_evening_person_extreme": "Morning/Evening Chronotype (extremes)",
    "nap_day_extreme": "Nap during day (extremes)",
    "doze_day": "Daytime Dozing",
    "snoring": "Snoring",
}


target_labels_xaxis = {
    "pheno_extreme": "Insomnia",
    "sleep_duration_extreme": "Sleep duration",
    "morning_extreme": "Easiness Getting up in the morning",
    "morning_evening_person_extreme": "Morning/Evening chronotype",
    "nap_day_extreme": "Daytime nap",
    "doze_day": "Daytime sleepiness",
    "snoring": "Snoring",
}

models = [
    "gset",
    "gsrf",
    "linearsvchc",
    "gslinearsvm",
    "logithc",
    "gssvm_rbf",
    "stacked_linearsvcheursiticc",
]


model_labels = {
    "linearsvm": "Linear SVM",
    "linearsvchc": "Linear SVM (Heuristic C)",
    "logithc": "Logit (Heuristic C)",
    "gsrf": "Random Forest",
    "gset": "Extra Trees",
    "optunasvm_rbf": "SVM (RBF Kernel)",
    "stacked_linearsvcheursiticc": "Stacked",
}

scores_labels = {
    "roc_auc": "ROC-AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1-Score",
    "average_precision": "Average Precision",
}

extreme_pheno_targets = {
    "sleep_duration_extreme": {  # 1160
        "target": "Sleep_duration-2.0",
        "extremes": "quantiles",
        "pos_labels": [3],
    },
    "morning_extreme": {  # 1170
        "target": "Getting_up_in_morning-2.0",
        "pos_labels": [1, 2],  # Not at all easy
        "extremes": [1, 2, 4],  # Not at all easy, Not very easy, Very easy
    },
    "morning_evening_person_extreme": {  # 1180
        "target": "Morning/evening_person_(chronotype)-2.0",
        "pos_labels": [4],  # Evening
        "extremes": [1, 4],  # Morning Evening
    },
    "nap_day_extreme": {  # 1190
        "target": "Nap_during_day-2.0",
        "pos_labels": [3],  # usually
        "extremes": [1, 3],  # never, usually
    },
    "pheno_extreme": {  # 1200
        "target": "Sleeplessness_/_insomnia-2.0",
        "pos_labels": [3],
        "extremes": [1, 3],
    },
    "snoring": {  # 1210
        "target": "Snoring-2.0",
        "pos_labels": [1],
        "extremes": [1, 2],  # Yes, No
    },
    "doze_day": {  # 1220
        "target": "Daytime_dozing_/_sleeping_(narcolepsy)-2.0",
        "pos_labels": [1, 2],
        "extremes": [0, 1, 2],  # Never, Sometimes, Often
    },
}


results_path = Path(__file__).parent / "results"


# %% Load data

validation_scores = {
    "target": [],
    "model": [],
    "balanced_accuracy": [],
    "f1": [],
    "average_precision": [],
    "roc_auc": [],
}

for t_model in models:
    for t_target in targets:
        target_path = results_path / t_target
        for fname in target_path.glob(f"{t_target}_{t_model}_validation.csv"):
            # print(f"Loading {fname}")
            t_df = pd.read_csv(fname, sep=";", index_col=0)
            y_true = t_df["y_true"]
            y_true = y_true.isin(
                extreme_pheno_targets[t_target]["pos_labels"]
            ).astype(int)
            y_pred = t_df["y_pred"]
            if "y_probas" not in t_df.columns:
                print(
                    f"No y_probas column for {t_target} / {t_model}, skipping"
                )
                y_probas = y_pred
            else:
                y_probas = t_df["y_probas"]
            t_roc_auc = roc_auc_score(y_true, y_probas)
            t_f1 = f1_score(y_true, y_pred)
            t_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            t_average_precision = average_precision_score(y_true, y_probas)
            validation_scores["target"].append(t_target)
            validation_scores["model"].append(t_model)
            validation_scores["balanced_accuracy"].append(t_balanced_accuracy)
            validation_scores["f1"].append(t_f1)
            validation_scores["average_precision"].append(t_average_precision)
            validation_scores["roc_auc"].append(t_roc_auc)

validation_df = pd.DataFrame(validation_scores)
# %%
validation_df.to_csv(
    results_path / "results_validation.csv", index=False, sep=";"
)

for t_col in ["balanced_accuracy", "f1", "average_precision", "roc_auc"]:
    validation_df[t_col] = validation_df[t_col].round(3)

# %%


def sort_func(a):
    if a[0] in targets:
        return a.map(lambda x: targets.index(x))
    else:
        return a.map(lambda x: models.index(x))


validation_df.sort_values(["target", "model"], key=sort_func, inplace=True)
# %%
validation_df.replace(
    {"target": target_labels_xaxis, "model": model_labels},
    inplace=True,
)
validation_df.rename(columns=scores_labels, inplace=True)
validation_df.to_csv(
    results_path / "results_validation_formatted.csv", index=False, sep=";"
)

# %%
