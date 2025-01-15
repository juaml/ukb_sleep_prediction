# %%

import json
from pathlib import Path

import pandas as pd
from mlconfound.stats import full_confound_test, partial_confound_test


# %% Config paths

results_path = Path(__file__).parent / "results"

# List of the best models, taken from the results of 8-create_results_table.py
with open(results_path / "best_models.json", "r") as f:
    best_models = json.load(f)


confounds = ["Sex-0.0", "AgeAtScan"]

# %% Get the target/model pairs to test

to_test = []

for _, t_models in best_models.items():
    for t_target, t_model in t_models.items():
        to_test.append((t_target, t_model))

to_test = set(to_test)


# %%

confound_test_results = {
    "target": [],
    "model": [],
    "test": [],
    "p_value": [],
    "p_ci": [],
    "r2_y_c": [],
    "r2_yhat_c": [],
    "r2_y_yhat": [],
    "expected_r2_yhat": [],
}

for t_target, t_model in to_test:
    t_results_path = results_path / t_target

    df_predictions = pd.read_csv(
        t_results_path / f"{t_target}_{t_model}_validation.csv",
        sep=";",
        index_col=0,
    )

    partial_test_result = partial_confound_test(
        y=df_predictions["y_true"].values,
        yhat=df_predictions["y_probas"].values,
        c=df_predictions[confounds[0]].values,
        num_perms=1000,
        mcmc_steps=50,
        cat_y=True,
        cat_c=True,
        cat_yhat=False,
        return_null_dist=True,
        random_state=42,
    )

    confound_test_results["target"].append(t_target)
    confound_test_results["model"].append(t_model)
    confound_test_results["test"].append("partial")
    confound_test_results["p_value"].append(partial_test_result.p)
    confound_test_results["p_ci"].append(partial_test_result.p_ci)
    confound_test_results["r2_y_c"].append(partial_test_result.r2_y_c)
    confound_test_results["r2_yhat_c"].append(partial_test_result.r2_yhat_c)
    confound_test_results["r2_y_yhat"].append(partial_test_result.r2_y_yhat)
    confound_test_results["expected_r2_yhat"].append(
        partial_test_result.expected_r2_yhat_c
    )

    full_test_result = full_confound_test(
        y=df_predictions["y_true"].values,
        yhat=df_predictions["y_probas"].values,
        c=df_predictions[confounds[0]].values,
        num_perms=1000,
        mcmc_steps=50,
        cat_y=True,
        cat_c=True,
        cat_yhat=False,
        return_null_dist=True,
        random_state=42,
    )

    confound_test_results["target"].append(t_target)
    confound_test_results["model"].append(t_model)
    confound_test_results["test"].append("full")
    confound_test_results["p_value"].append(full_test_result.p)
    confound_test_results["p_ci"].append(full_test_result.p_ci)
    confound_test_results["r2_y_c"].append(full_test_result.r2_y_c)
    confound_test_results["r2_yhat_c"].append(full_test_result.r2_yhat_c)
    confound_test_results["r2_y_yhat"].append(full_test_result.r2_y_yhat)
    confound_test_results["expected_r2_yhat"].append(
        full_test_result.expected_r2_y_yhat
    )

confounds_test_df = pd.DataFrame(confound_test_results)

# %% sort and format


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

target_labels_xaxis = {
    "pheno_extreme": "Insomnia",
    "sleep_duration_extreme": "Sleep duration",
    "morning_extreme": "Getting up in the morning",
    "morning_evening_person_extreme": "Morning/Evening chronotype",
    "nap_day_extreme": "Daytime nap",
    "doze_day": "Daytime sleepiness",
    "snoring": "Snoring",
}


model_labels = {
    "gslinearsvm": "GSLinearSVM",
    "linearsvchc": "LinearSVMHC",
    "logithc": "LogitHC",
    "gsrf": "GSRF",
    "gset": "GSET",
    "gssvm_rbf": "GSSVM-RBF",
    "stacked_linearsvcheursiticc": "Stacked",
}


def sort_func(a):
    if a[0] in targets:
        return a.map(lambda x: targets.index(x))
    else:
        return a.map(lambda x: models.index(x))


confounds_test_df.sort_values(["target", "model"], key=sort_func, inplace=True)
# %%
confounds_test_df.replace(
    {"target": target_labels_xaxis, "model": model_labels},
    inplace=True,
)
confounds_test_df.to_csv(
    results_path / "confounds_test.csv", index=False, sep=";"
)
