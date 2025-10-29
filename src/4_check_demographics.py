# %%
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


lib_dir = Path(__file__).parent.parent / "lib"
print(f"Adding {lib_dir} to path")
sys.path.append(lib_dir.as_posix())
from nimrls.io import read_features_jay, read_pheno, read_prs

# %%
results_dir = Path(__file__).parent / "results_confounds"

# %%
start_time = time.time()

targets = [
    "pheno_extreme",
    "sleep_duration_extreme",
    "morning_extreme",
    "morning_evening_person_extreme",
    "nap_day_extreme",
    "snoring",
    "doze_day",
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

# %%
data = features.copy()
data = data.join(pheno_data_df, how="inner")
data.columns = data.columns.astype(str)

sub_ids = np.array(data.index)

# %%
np.savetxt(data_dir / "subjects_list.txt", sub_ids, fmt="%s")

# %%


def cohen_d(x, y):
    """Compute Cohen's d for effect size."""
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    s = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / s


target_demographics = {
    "target": [],
    "n_positive": [],
    "n_negative": [],
    "n_total": [],
    "age_mean": [],
    "age_mean_positive": [],
    "age_mean_negative": [],
    "age_std": [],
    "age_std_positive": [],
    "age_std_negative": [],
    "age-stats_t-stat": [],
    "age-stats_p-val": [],
    "age-stats_cohen-d": [],
    "sex_male": [],
    "sex_female": [],
    "sex_ratio_positive": [],
    "sex_ratio_negative": [],
    "sex-stats_chi2-stat": [],
    "sex-stats_p-val": [],
    "sex-stats_phi": [],
}
for target in targets:
    data = features.copy()
    if target in extreme_pheno_targets.keys() or target == "both_extreme":
        data = data.join(pheno_data_df, how="inner")
    if target in ["prs_extreme", "both_extreme"]:
        data = data.join(geno_data_df, how="inner")
    # data = data.join(apoe_data_df, how="inner")

    data.columns = data.columns.astype(str)

    if target in extreme_pheno_targets.keys() or target in [
        "prs_extreme",
        "both_extreme",
    ]:
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
                data = data[
                    data[target].isin(
                        extreme_pheno_targets[target]["extremes"]
                    )
                ]
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
            data.rename(
                {"pheno_extreme": "both_extreme"}, axis=1, inplace=True
            )
            y = "both_extreme"
            extra_params = {"pos_labels": [1]}
    pos_labels = extra_params["pos_labels"]
    n_pos = data[y].isin(pos_labels).sum()
    n_neg = data.shape[0] - n_pos
    target_demographics["target"].append(target)
    target_demographics["n_positive"].append(n_pos)
    target_demographics["n_negative"].append(n_neg)
    target_demographics["n_total"].append(data.shape[0])
    target_demographics["age_mean"].append(data["AgeAtScan"].mean())
    target_demographics["age_std"].append(data["AgeAtScan"].std())
    
    x = data.loc[data[y].isin(pos_labels), "AgeAtScan"]
    z = data.loc[~data[y].isin(pos_labels), "AgeAtScan"]
    t_stat, p_val = stats.ttest_ind(
        x,
        z,
        equal_var=False,
    )
    target_demographics["age_mean_positive"].append(x.mean())
    target_demographics["age_mean_negative"].append(z.mean())
    target_demographics["age_std_positive"].append(x.std())
    target_demographics["age_std_negative"].append(z.std())

    target_demographics["age-stats_cohen-d"].append(cohen_d(x, z))
    target_demographics["age-stats_t-stat"].append(t_stat)
    target_demographics["age-stats_p-val"].append(p_val)

    # Sex stats
    target_demographics["sex_male"].append(data["Sex-0.0"].sum())
    target_demographics["sex_female"].append((data["Sex-0.0"] == 0).sum())

    x = data.loc[data[y].isin(pos_labels), "Sex-0.0"]
    z = data.loc[~data[y].isin(pos_labels), "Sex-0.0"]

    target_demographics["sex_ratio_positive"].append(x.sum() / (x == 0).sum())
    target_demographics["sex_ratio_negative"].append(z.sum() / (z == 0).sum())

    contingency_table = pd.crosstab(
        data[y].isin(pos_labels), data["Sex-0.0"]
    )

    chi2, p_val, dof, ex = stats.chi2_contingency(contingency_table)
    phi = np.sqrt(chi2 / data.shape[0])

    target_demographics["sex-stats_phi"].append(phi)
    target_demographics["sex-stats_chi2-stat"].append(chi2)
    target_demographics["sex-stats_p-val"].append(p_val)
    print(target)
    print(data[y].isin(pos_labels).value_counts())

df_demographics = pd.DataFrame(target_demographics)

# %%
df_demographics.to_csv(
    results_dir / "demographics_summary.csv", sep=";", index=False
)

# %% all demographics
data = features.copy()
data = data.join(pheno_data_df, how="inner")
data.columns = data.columns.astype(str)
# %%
demographics = [
    "Sex-0.0",
    "AgeAtScan",
    "YearsOfEducation",
    "Overall_health_rating-0.0",
    "Ethnic_background-0.0"
]


# %%

health_coding = {
    1: "Excellent",
    2: "Good",
    3: "Fair",
    4: "Poor",
    -1: "Do not know",
    -3: "Prefer not to answer",
}

data["Overall_health_rating-0.0"] = data["Overall_health_rating-0.0"].map(health_coding)

# %%
ethnicity_coding = {
    1: "White",
    1001: "White", #"British",
    1002: "White", #"Irish",
    1003: "White", #"Any other white background",
    2: "Mixed",
    2001: "Mixed", #"White and Black Caribbean",
    2002: "Mixed", #"White and Black African",
    2003: "Mixed", #"White and Asian",
    2004: "Mixed", #"Any other mixed background",
    3: "Asian or Asian British",
    3001: "Asian or Asian British", #"Indian",
    3002: "Asian or Asian British", #"Pakistani",
    3003: "Asian or Asian British", #"Bangladeshi",
    3004: "Asian or Asian British", #"Any other Asian background",
    4: "Black or Black British",
    4001: "Black or Black British", #"Caribbean",
    4002: "Black or Black British", #"African",
    4003: "Black or Black British", #"Any other Black background",
    5: "Chinese",
    6: "Other ethnic group",
    -1: "Do not know",
    -3: "Prefer not to answer",
}

data["Ethnic_background-0.0"] = data["Ethnic_background-0.0"].map(ethnicity_coding)

# %%

print(data[demographics].describe())
print("-----")
print("Sex")
print(
    f"Male {data[demographics[0]].sum()} - Female {(data[demographics[0]] == 0).sum()}"
)
# %%
print("-----")
print("Ethnicity")
print(
    data.groupby("Ethnic_background-0.0")["AgeAtScan"].count()
)
# %%
print("-----")
print("Self-reported health rating  ")
print(
    data.groupby("Overall_health_rating-0.0")["AgeAtScan"].count()
)
# %%
