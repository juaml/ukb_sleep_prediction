# %%
from pathlib import Path
import time
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import sys
lib_dir = Path(__file__).parent.parent / "lib"
print(f"Adding {lib_dir} to path")
sys.path.append(lib_dir.as_posix())
from nimrls.io import read_prs, read_pheno, read_features_jay


# %%
start_time = time.time()

targets = [
    "pheno_extreme",
    "sleep_duration_extreme",
    "morning_extreme",
    "morning_evening_person_extreme",
    "nap_day_extreme",
    "snoring",
    "doze_day"
]

# Directories

data_dir = Path("/data/project/ukb_rls/data/features")

geno_data_fname = data_dir / "rls_Schormair_etal_PRS.tsv"
# geno_data_fname = data_dir / "bmi_PGS000034_PRS.tsv"
pheno_data_fname = data_dir / "phenotypes.csv"
# apoe_data_fname = data_dir / "APOE.tsv"


# Read data
geno_data_df = read_prs(geno_data_fname)
pheno_data_df = read_pheno(pheno_data_fname)
# apoe_data_df = read_apoe(apoe_data_fname)

features = read_features_jay(data_dir)

extreme_pheno_targets = {
    "sleep_duration_extreme": {  # 1160
        "target": "Sleep_duration-2.0",
        "extremes": "quantiles",
    },
    "morning_extreme": {  # 1170
        "target": "Getting_up_in_morning-2.0",
        "pos_labels": [1, 2],  # Not at all easy
        "extremes": [1, 2, 4], # Not at all easy, Not very easy, Very easy
    },
    "morning_evening_person_extreme": {  # 1180
        "target": "Morning/evening_person_(chronotype)-2.0",
        "pos_labels": [4],  # Evening
        "extremes": [1, 4], # Morning Evening
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

# %%
data = features.copy()
data = data.join(pheno_data_df, how="inner")
data.columns = data.columns.astype(str)

sub_ids = np.array(data.index)

np.savetxt(data_dir / "subjects_list.txt", sub_ids, fmt="%s")

# %%
for target in targets:
    data = features.copy()
    if target in extreme_pheno_targets.keys() or target == "both_extreme":
        data = data.join(pheno_data_df, how="inner")
    if target in ["prs_extreme", "both_extreme"]:
        data = data.join(geno_data_df, how="inner")
    # data = data.join(apoe_data_df, how="inner")

    data.columns = data.columns.astype(str)


    if target in extreme_pheno_targets.keys() or target in ["prs_extreme", "both_extreme"]:
        if target in extreme_pheno_targets:
            t_target = extreme_pheno_targets[target]["target"]
            data = data[~data[t_target].isna()]

            if extreme_pheno_targets[target]["extremes"] == "quantiles":
                data = data[data[t_target] > 0]
                quantiles = np.quantile(data[t_target], [0.25, 0.5, 0.75])
                pheno_quantiles = np.digitize(data[t_target], quantiles)
                data[target] = pheno_quantiles
                print(f"Quantiles: {quantiles}")
                print(f"Dist: {data[target].value_counts()}")
                data = data[data[target].isin([0, 3])]
                extra_params = {"pos_labels": [3]}
            else:
                data[target] = data[t_target].astype(int)
                data = data[data[target].isin(
                    extreme_pheno_targets[target]["extremes"]
                )]
                extra_params = {
                    "pos_labels": extreme_pheno_targets[target]["pos_labels"]
                }
            y = target
        elif target == "prs_extreme":
            t_target = "prs"
            data = data[~data[t_target].isna()]
            quantiles = np.quantile(data[t_target], [0.25, 0.5, 0.75])
            pheno_quantiles = np.digitize(data[t_target], quantiles)
            data["prs_extreme"] = pheno_quantiles
            data = data[data["prs_extreme"].isin([0, 3])]
            extra_params = {"pos_labels": [3]}
            y = "prs_extreme"
        elif target == "both_extreme":
            # Get PRS extremes
            data = data[~data["prs"].isna()]
            quantiles = np.quantile(data["prs"], [0.25, 0.5, 0.75])
            pheno_quantiles = np.digitize(data["prs"], quantiles)
            data["prs_extreme"] = pheno_quantiles
            data = data[data["prs_extreme"].isin([0, 3])]
            data["prs_extreme"] = (data["prs_extreme"] == 3).astype(int)
            # Get insomnia extremes
            t_target = "Sleeplessness_/_insomnia-2.0"
            data = data[~data[t_target].isna()]
            data["pheno_extreme"] = data[t_target].astype(int)
            data = data[data["pheno_extreme"].isin([1, 3])]
            data["pheno_extreme"] = (data["pheno_extreme"] == 3).astype(int)
            # Get intersection
            data = data[data["pheno_extreme"] == data["prs_extreme"]]
            data.drop(["prs_extreme"], axis=1, inplace=True)
            data.rename({"pheno_extreme": "both_extreme"}, axis=1, inplace=True)
            y = "both_extreme"
            extra_params = {"pos_labels": [1]}
    pos_labels = extra_params["pos_labels"]
    print(target)
    print(data[y].isin(pos_labels).value_counts())


# %% all demographics
data = features.copy()
data = data.join(pheno_data_df, how="inner")
data.columns = data.columns.astype(str)
# %%
demographics = [
    "Sex-0.0",
    "AgeAtScan",
]
print(data[demographics].describe())
print("-----")
print("Sex")
print(f"Male {data[demographics[0]].sum()} - Female {(data[demographics[0]] == 0).sum()}")
# %%
