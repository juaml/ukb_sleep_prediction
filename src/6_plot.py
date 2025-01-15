# %%
from pathlib import Path
import pandas as pd
import numpy as np
from julearn.viz import plot_scores

# %%
target = "pheno_extreme"
results_path = Path("./results")
# %%
all_df = []
pattern = f"{target}_*_cv_scores_*.csv"
for fname in results_path.glob(pattern):
    print(fname)
    if "brainonly" in fname.stem or "cognitive" in fname.stem:
        continue
    t_df = pd.read_csv(fname, sep=";")
    model_name = fname.stem.replace(f"{target}_", "").split("_cv_scores")[0]
    t_df["model"] = model_name
    t_df["cv_mdsum"] = "same"
    all_df.append(t_df)
if len(all_df) == 0:
    raise ValueError("No files found")


# Now group the scores by model name (in case folds are passed separately)
model_names = []
for t_score in all_df:
    model_names.extend(t_score["model"].unique())
model_names = np.unique(model_names)

grouped_scores = []

for t_name in model_names:
    t_scores = []
    for t_score in all_df:
        if t_name in t_score["model"].unique():
            t_scores.append(t_score[t_score["model"] == t_name])
    grouped_scores.append(pd.concat(t_scores))

plot = plot_scores(*grouped_scores)

plot.servable()