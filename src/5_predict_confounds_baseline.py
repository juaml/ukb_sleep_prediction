# %%
import logging
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import joblib
from joblib_htcondor import register_htcondor
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.svm import LinearSVC

import julearn
from julearn import run_cross_validation
from julearn.config import set_config
from julearn.pipeline import PipelineCreator


lib_dir = Path(__file__).parent.parent / "lib"
print(f"Adding {lib_dir} to path")
sys.path.append(lib_dir.as_posix())
from nimrls.io import read_pheno  # noqa: E402
from nimrls.logging import (  # noqa: E402
    configure_logging,
    log_versions,
    logger,
    raise_error,
)
from nimrls.ml import (  # noqa: E402
    LinearSVCHeuristicC,
    LogisticRegressionHeuristicC,
)


configure_logging()
julearn.utils.logging.configure_logging("INFO")
log_versions()

register_htcondor()

set_config("disable_x_verbose", True)
set_config("disable_xtypes_verbose", True)
set_config("disable_xtypes_check", True)
set_config("disable_x_check", True)

start_time = time.time()

# Shared data directory for joblib
shared_data_dir = Path("/data/group/ukb_rls/joblib_htcondor_shared_confounds")
shared_data_dir.mkdir(parents=True, exist_ok=True)

parser = ArgumentParser(description="Run the predictive models.")
parser.add_argument(
    "--target",
    metavar="target",
    type=str,
    help="Target to predict",
    required=True,
)
parser.add_argument(
    "--model",
    metavar="model",
    type=str,
    help="Model to use",
    required=True,
)

parser.add_argument(
    "--data", metavar="data", type=str, help="Path to data", default="../data"
)
args = parser.parse_args()

# target = "pheno_extreme"
# model_name = "linearsvchc"
target = args.target
model_name = args.model
data_dir = Path(args.data)

out_dir = Path(__file__).parent / "phenotype_results" / target
out_dir.mkdir(parents=True, exist_ok=True)


X = ["Sex-0.0", "AgeAtScan"]

N_REPEATS = 5
N_SPLITS = 5

predict_proba = False

pheno_data_fname = data_dir / "phenotypes.csv"

pheno_data_df = read_pheno(pheno_data_fname)

data = pheno_data_df

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
sub_ids = np.loadtxt(data_dir / "subjects_list.txt", dtype=str)
data = data.loc[sub_ids]

# %%

t_target = extreme_pheno_targets[target]["target"]
data = data[~data[t_target].isna()]

if extreme_pheno_targets[target]["extremes"] == "quantiles":
    data = data[data[t_target] > 0]
    quantiles = np.quantile(data[t_target], [0.25, 0.5, 0.75])
    pheno_quantiles = np.digitize(data[t_target], quantiles)
    data[target] = pheno_quantiles
    logger.info(f"Quantiles: {quantiles}")
    logger.info(f"Dist: {data[target].value_counts()}")
    data = data[data[target].isin([0, 3])]
    extra_params = {"pos_labels": [3]}
else:
    data[target] = data[t_target].astype(int)
    data = data[
        data[target].isin(extreme_pheno_targets[target]["extremes"])
    ]
    extra_params = {
        "pos_labels": extreme_pheno_targets[target]["pos_labels"]
    }
y = target
# %%
cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
)

scoring = [
    "roc_auc",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
]

creator = PipelineCreator(problem_type="classification", apply_to="*")
creator.add("zscore")
search_params = {}
if model_name in ["rf", "et"]:
    creator.add(model_name)
elif model_name == "svm":
    creator.add(model_name, probability=True)
elif model_name == "gssvm":
    creator.add(
        "svm",
        C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        kernel="linear",
        probability=True,
    )
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}
elif model_name == "gssvm_rbf":  # Used
    creator.add(
        "svm",
        C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        kernel="rbf",
        gamma=[
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            10,
            100,
            1000,
        ],
        probability=True,
    )
    predict_proba = True
    n_jobs = -1
    search_params = {
        "kind": "grid",
        "scoring": "balanced_accuracy",
        "pre_dispatch": "all",
    }
elif model_name == "gsrf":  # Used
    n_estimators = [200, 500]
    criterion = ["gini", "entropy", "log_loss"]
    max_features = ["sqrt", "log2"]
    creator.add(
        "rf",
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=max_features,
    )
    predict_proba = True
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}
elif model_name == "gset":  # Used
    n_estimators = [200, 500]
    max_features = ["sqrt", "log2"]
    criterion = ["gini", "entropy", "log_loss"]
    creator.add(
        "et",
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=max_features,
    )
    predict_proba = True
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}
elif model_name == "gslinearsvm":
    model = LinearSVC()
    creator.add(
        model,
        name="linearsvc",
        C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        dual=False,
    )
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}
    n_jobs = -1
    search_params = {
        "kind": "grid",
        "scoring": "balanced_accuracy",
        "pre_dispatch": "all",
    }
elif model_name == "linearsvm":  # Used
    model = LinearSVC()
    creator.add(
        model,
        name="linearsvc",
        C=0.001,
        dual=False,
        penalty="l1",
    )
elif model_name == "linearsvchc":  # Used
    model = LinearSVCHeuristicC()
    creator.add(
        model,
        name="linearsvcheuristicc",
        dual=False,
        penalty="l1",
    )
    predict_proba = True
elif model_name == "logithc":  # Used
    model = LogisticRegressionHeuristicC()
    creator.add(
        model,
        name="logithc",
        dual=False,
        penalty="l1",
        solver="liblinear",
    )
    predict_proba = True
elif model_name == "stacked_linearsvcheursiticc":  # Used
    feature_types = ["GMD", "Surface", "fALFF", "GCOR", "LCOR"]
    models = []
    for t_ftype in feature_types:
        # t_regexp = f"{t_prefix}_.*"
        t_model = PipelineCreator(
            problem_type="classification", apply_to=t_ftype
        )
        t_model.add("filter_columns", apply_to="*", keep=t_ftype)
        t_model.add("zscore")
        t_model.add(
            LinearSVCHeuristicC(),
            name="linearsvcheuristicc",
            dual=False,
            penalty="l1",
        )
        models.append((f"model_{t_ftype}", t_model))
    creator = PipelineCreator(problem_type="classification")
    creator.add(
        "stacking",
        estimators=[models],
        apply_to="*",
    )
    predict_proba = True
elif model_name == "dummy":
    creator.add("dummy")
    predict_proba = True
elif model_name == "dummy_stratified":
    creator.add("dummy", strategy="stratified")
    predict_proba = True
elif model_name == "optunasvm_rbf":  # Used
    creator.add(
        "svm",
        C=(0.0001, 10000, "log-uniform"),
        kernel="rbf",
        gamma=(1e-7, 1000, "log-uniform"),
        probability=True,
    )
    predict_proba = True
    search_params = {
        "kind": "optuna",
        "scoring": "balanced_accuracy",
        "n_trials": 50,
    }
elif model_name == "optunasvm":  # Used
    creator.add(
        "svm",
        C=(0.0001, 10000, "log-uniform"),
        kernel="linear",
        probability=True,
    )
    predict_proba = True
    search_params = {
        "kind": "optuna",
        "scoring": "balanced_accuracy",
        "n_trials": 50,
    }
else:
    raise_error(f"Unknown model {model_name} for {target}")

logger.info(f"TOTAL samples = {data.shape[0]}")
logger.info(f"TOTAL features + target = {data.shape[1]}")
logger.info("Filtering out non-relevant columns")
t_data = data[[*X, y]]
logger.info("Dropping NaNs")
t_data = t_data.dropna()

logger.info("Splitting data into train and validation")
train_data, validate_data = train_test_split(t_data, test_size=0.3)

logger.info(f"TRAIN samples = {train_data.shape[0]}")
logger.info(f"TRAIN features + target = {train_data.shape[1]}")

logger.info(f"VALIDATE samples = {validate_data.shape[0]}")
logger.info(f"VALIDATE features + target = {validate_data.shape[1]}")


X_types = {"continuous": X}

# %%
with joblib.parallel_config(
    backend="htcondor",
    pool="head2.htc.inm7.de",
    n_jobs=-1,
    request_cpus=1,
    request_memory="16GB",
    request_disk="1GB",
    verbose=1000,
    throttle=[200, 550],  # each job needs 1Mb. 26 CV * 550 * 1 Mb = 13.96 Gb
    shared_data_dir=shared_data_dir,
    worker_log_level=logging.DEBUG,
    poll_interval=5,
    max_recursion_level=1,  # Outer + Inner CV, no more than that
):
    out = run_cross_validation(
        X=X,
        y=y,
        X_types=X_types,
        data=train_data,
        cv=cv,
        model=creator,
        return_estimator="cv",
        return_inspector=True,
        return_train_score=True,
        scoring=scoring,
        search_params=search_params,
        **extra_params,
    )
# %%

scores, inspector = out

fname = f"phenotype_{target}_{model_name}_cv_scores.csv"

logger.info(f"Saving CV scores to {fname}")
scores.to_csv(out_dir / fname, sep=";")


elapsed_time = time.time() - start_time
logger.info(
    "Elapsed CV time {}".format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    )
)

logger.info("Predicting fold probabilities")
try:
    fold_predictions = inspector.folds.predict_proba()
    fname = (
        f"phenotype_{target}_{model_name}"
        f"_fold_predictions.csv"
    )
    fold_predictions.to_csv(out_dir / fname, sep=";")
except Exception as e:  # noqa: BLE001
    logger.error(e)


logger.info(f"Saving validation results to {fname}")
elapsed_time = time.time() - start_time
logger.info(
    "Elapsed time {}".format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    )
)
logger.info("Done!")
