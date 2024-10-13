import logging
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import joblib
import julearn
import numpy as np
import optuna
import pandas as pd
from joblib_htcondor import logging as htcondor_logging
from joblib_htcondor import register_htcondor
from julearn import run_cross_validation
from julearn.config import set_config
from julearn.pipeline import PipelineCreator
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.svm import LinearSVC

lib_dir = Path(__file__).parent.parent / "lib"
print(f"Adding {lib_dir} to path")
sys.path.append(lib_dir.as_posix())
from nimrls.io import read_features_jay, read_pheno, read_prs  # noqa: E402
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
htcondor_logging.configure_logging("INFO")
log_versions()

register_htcondor()

set_config("disable_x_verbose", True)
set_config("disable_xtypes_verbose", True)
set_config("disable_xtypes_check", True)
set_config("disable_x_check", True)

start_time = time.time()

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
    "--features",
    metavar="features",
    type=str,
    help="Features to use",
    default=None,
    required=False,
)
parser.add_argument(
    "--optionals",
    metavar="optionals",
    type=str,
    help="optionals",
    required=False,
    default=None,
    nargs="*",
)
parser.add_argument(
    "--data", metavar="data", type=str, help="Path to data", default="../data"
)
parser.add_argument(
    "--fold",
    metavar="fold",
    type=int,
    help="Fold to compute. If None, compute all of them",
    default=None,
)
args = parser.parse_args()

N_REPEATS = 5
N_SPLITS = 5

IS_DEBUG_TEST = False

n_jobs = 1
ht_condor_recursion_level = 0
throttle = N_REPEATS * N_SPLITS

target = args.target
model_name = args.model
features_name = args.features
fold = args.fold
optionals = args.optionals

logger.info(f"Optionals: {optionals}")

if optionals is None:
    optionals = []

if not isinstance(optionals, list):
    optionals = [optionals]

if fold is None:
    logger.info("Computing all folds")
    n_jobs = -1
    ht_condor_recursion_level = 0
else:
    logger.info(f"Computing fold {fold}")
# Directories

data_dir = Path(args.data)

geno_data_fname = data_dir / "rls_Schormair_etal_PRS.tsv"
# geno_data_fname = data_dir / "bmi_PGS000034_PRS.tsv"
pheno_data_fname = data_dir / "phenotypes.csv"
# apoe_data_fname = data_dir / "APOE.tsv"

# Shared data directory for joblib
shared_data_dir = Path("/data/group/riseml/joblib_htcondor_shared")
shared_data_dir.mkdir(parents=True, exist_ok=True)

# Read data
geno_data_df = read_prs(geno_data_fname)
pheno_data_df = read_pheno(pheno_data_fname)
# apoe_data_df = read_apoe(apoe_data_fname)

features = read_features_jay(data_dir)

cognitive_features = [
    "20016-2.0",
    "20023-2.0",
    "4282-2.0",
    "6348-2.0",
    "6350-2.0",
    "399-2.1",
    "399-2.2",
    "400-2.1",
    "400-2.2",
    "20018-2.0",
    "23323-2.0",
    "23324-2.0",
    "4526-2.0",
    "4559-2.0",
    "4537-2.0",
    "4548-2.0",
    "4570-2.0",
    "4581-2.0",
    "20458-0.0",
    "20459-0.0",
    "20460-0.0",
    "neuroticism_score",
    "anxiety_score",
    "depression_score_no_sleep",
    "CIDI_score_no_sleep",
]

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

if "cognitive" in optionals:
    logger.info("Adding cognitive scores data")
    cognitive_fname = data_dir / "All_MRI_Cognitive_scores_without_sleep.csv"
    cognitive_df = pd.read_csv(cognitive_fname, sep=",")
    cognitive_df = cognitive_df.set_index("eid")
    cognitive_df.index = [f"sub-{x}" for x in cognitive_df.index]  # type: ignore
    cognitive_df.index.name = "SubjectID"  # type: ignore

    cognitive_df = cognitive_df[cognitive_features]
    cognitive_df = cognitive_df.add_prefix("cognitive_")
    cognitive_df = cognitive_df.dropna()
    logger.info(f"Cognitive samples {cognitive_df.shape[0]}")
    logger.info(f"Cognitive features {cognitive_df.shape[1]}")
    # cognitive_df = cognitive_df.set_index("eid_0")
    features = features.join(cognitive_df, how="inner")


X = list(features.columns)

if "brainonly" in optionals:
    logger.info("Using only brain features")
    X = [x for x in features.columns if x not in cognitive_features]
if "cognitive-only" in optionals:
    logger.info("Using only cognitive features")
    X = cognitive_features

data = features

if target in extreme_pheno_targets.keys() or target == "both_extreme":
    data = data.join(pheno_data_df, how="inner")
if target in ["prs_extreme", "both_extreme"]:
    data = data.join(geno_data_df, how="inner")
# data = data.join(apoe_data_df, how="inner")

data.columns = data.columns.astype(str)

predict_proba = False
X_types = {
    "GMD": "GMD_.*",
    "Surface": "Surface_.*",
    "fALFF": "fALFF_.*",
    "GCOR": "GCOR_.*",
    "LCOR": "LCOR.*",
}
if "cognitive" in optionals:
    X_types["cognitive"] = "cognitive_.*"

if IS_DEBUG_TEST:
    # FOR TESTING: use only 10% of the samples
    logger.info("Using 5% of the samples")
    data = data.sample(frac=0.05)

if target == "prs":
    raise NotImplementedError("PRS not implemented yet")
    # y = "prs"
    # if model_name == "ridge":
    #     jumodel = "ridge"
    # elif model_name == "svm":
    #     jumodel = "svm"
    # else:
    #     raise_error(f"Unknown model {model_name} for {target}")
    # preprocess_X = "zscore"
    # problem_type = "regression"
    # scoring = ["neg_mean_absolute_error", "r2"]
    # cv = None
    # model_params = {}
    # extra_params = {}
elif target in extreme_pheno_targets.keys() or target in [
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
            logger.info(f"Quantiles: {quantiles}")
            logger.info(f"Dist: {data[target].value_counts()}")
            data = data[data[target].isin([0, 3])]
            extra_params = {"pos_labels": [3]}
        else:
            data[target] = data[t_target].astype(int)
            data = data[data[target].isin(extreme_pheno_targets[target]["extremes"])]
            extra_params = {"pos_labels": extreme_pheno_targets[target]["pos_labels"]}
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

    # Binary classification, do RepeatedStratifiedKFold
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
    search_params = None
    if model_name in ["rf", "et"]:
        creator.add(model_name)
    elif model_name == "svm":
        creator.add(model_name, probability=True)
        predict_proba = "proba"
    elif model_name == "gssvm":
        creator.add(
            "svm",
            C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
            kernel="linear",
            probability=True,
        )
        search_params = {"kind": "grid", "scoring": "balanced_accuracy"}
        n_jobs = -1
        ht_condor_recursion_level = 1
        # each job needs 0.6Gb. 26 CV * 10 * 5 * 0.6 Gb = 780 Gb
        throttle = [26, 50]
        predict_proba = "proba"
    elif model_name == "gssvm_rbf":  # Used
        if IS_DEBUG_TEST:
            creator.add(
                "svm",
                C=[0.1, 1, 10],
                kernel="rbf",
                gamma=[
                    1e-1,
                    1,
                    10,
                ],
                probability=True,
            )
        else:
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
        predict_proba = "proba"
        n_jobs = -1
        ht_condor_recursion_level = 1
        # each job needs 0.6Gb. 26 CV * 110 * 5 * 0.6 Gb = 8 Tb
        # Thottle to 60 to keep under 1 Tb
        throttle = [26, 60]
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
            n_jobs=1,
        )
        predict_proba = "proba"
        n_jobs = -1
        ht_condor_recursion_level = 1
        # each job needs 0.6Gb. 26 CV * 12 * 5 * 0.6 Gb = 936 Gb
        throttle = [26, 60]
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
            n_jobs=1,
        )
        predict_proba = "proba"
        n_jobs = -1
        ht_condor_recursion_level = 1
        # each job needs 0.6Gb. 26 CV * 12 * 5 * 0.6 Gb = 936 Gb
        throttle = [26, 60]
        search_params = {"kind": "grid", "scoring": "balanced_accuracy"}
    elif model_name == "gslinearsvm":
        model = LinearSVC()
        creator.add(
            model,
            name="linearsvc",
            C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        )
        predict_proba = "decision"
        n_jobs = -1
        ht_condor_recursion_level = 1
        # each job needs 0.6Gb. 26 CV * 10 * 5 * 0.6 Gb = 780 Gb
        throttle = [26, 50]
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
        predict_proba = "decision"
    elif model_name == "linearsvchc":  # Used
        model = LinearSVCHeuristicC()
        creator.add(
            model,
            name="linearsvcheuristicc",
            dual=False,
            penalty="l1",
        )
        n_jobs = 1
        predict_proba = "decision"
    elif model_name == "logithc":  # Used
        model = LogisticRegressionHeuristicC()
        creator.add(
            model,
            name="logithc",
            dual=False,
            penalty="l1",
            solver="liblinear",
        )
        n_jobs = 1
        predict_proba = "decision"
    elif model_name == "stacked_linearsvcheursiticc":  # Used
        feature_types = ["GMD", "Surface", "fALFF", "GCOR", "LCOR"]
        if "cognitive" in optionals:
            feature_types += ["cognitive"]
        models = []
        for t_ftype in feature_types:
            # t_regexp = f"{t_prefix}_.*"
            t_model = PipelineCreator(problem_type="classification", apply_to=t_ftype)
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
        n_jobs = 1
        predict_proba = "proba"
    elif model_name == "dummy":
        creator.add("dummy")
        predict_proba = "proba"
        n_jobs = 1
    elif model_name == "dummy_stratified":
        creator.add("dummy", strategy="stratified")
        predict_proba = "proba"
        n_jobs = 1
    elif model_name == "optunasvm_rbf":  # Used
        creator.add(
            "svm",
            C=(0.0001, 10000, "log-uniform"),
            kernel="rbf",
            gamma=(1e-7, 1000, "log-uniform"),
            probability=True,
        )
        predict_proba = "proba"
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
        predict_proba = "proba"
        search_params = {
            "kind": "optuna",
            "scoring": "balanced_accuracy",
            "n_trials": 50,
        }
    else:
        raise_error(f"Unknown model {model_name} for {target}")
else:
    raise_error(f"Unknown target {target}")


logger.info(f"TOTAL samples = {data.shape[0]}")
logger.info(f"TOTAL features + target = {data.shape[1]}")
logger.info("Filtering out non-relevant columns")
t_data = data[[*X, y]]
logger.info("Dropping NaNs")
t_data = t_data.dropna()

logger.info("Splitting data into train and validation")
train_data, validate_data = train_test_split(t_data, test_size=0.3, random_state=22)

logger.info(f"TRAIN samples = {train_data.shape[0]}")
logger.info(f"TRAIN features + target = {train_data.shape[1]}")

logger.info(f"VALIDATE samples = {validate_data.shape[0]}")
logger.info(f"VALIDATE features + target = {validate_data.shape[1]}")


return_estimator = "all"
if fold is not None and fold != 0:
    return_estimator = "cv"

if fold is not None:
    all_folds = list(cv.split(train_data, train_data[y]))
    cv = [all_folds[fold]]

suffix = ""
if fold is not None:
    suffix = f"_{fold}"

optionals_suffix = ""
if optionals is not None and len(optionals) > 0:
    optionals_suffix = "_" + "_".join(optionals)

if features_name is not None:
    X_julearn = f"{features_name}_.*"
    logger.info(f"Using features: {X}")
    optionals_suffix = f"{optionals_suffix}_{features_name}"
    X_types = {features_name: X_types[features_name]}
else:
    X_julearn = [f"{x}.*" for x in ["GMD", "Surface", "fALFF", "GCOR", "LCOR"]]

if IS_DEBUG_TEST:
    X_julearn = X_julearn[0]
    X_types = {k: v for k, v in X_types.items() if k == "GMD"}
    logger.info(f"Using features: {X_julearn} (DEBUG TEST)")
    logger.info(f"Using X_types: {X_types} (DEBUG TEST)")

logger.info(f"Suffix: {suffix}")
logger.info(f"Optionals suffix: {optionals_suffix}")

out_dir = Path(__file__).parent / "results" / target
out_dir.mkdir(parents=True, exist_ok=True)

# Set up study for optuna
if search_params is not None:
    if search_params["kind"] == "optuna":
        study_fname = f"{target}_{model_name}{optionals_suffix}_optuna{suffix}.db"
        this_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True,
            storage=f"sqlite:///{out_dir}/{study_fname}",
            study_name=f"{model_name}_{target}{optionals_suffix}{suffix}",
        )
        search_params["study"] = this_study


logger.info("Running cross-validation")
logging.getLogger("joblib_htcondor.backend").setLevel(logging.INFO)

with joblib.parallel_config(
    backend="htcondor",
    pool="head2.htc.inm7.de",
    n_jobs=n_jobs,
    request_cpus=1,
    request_memory="16GB",
    request_disk="1GB",
    verbose=1000,
    throttle=throttle,
    shared_data_dir=shared_data_dir,
    worker_log_level=logging.DEBUG,
    poll_interval=5,
    max_recursion_level=ht_condor_recursion_level,
):
    out = run_cross_validation(
        X=X_julearn,
        y=y,
        X_types=X_types,
        data=train_data,
        cv=cv,
        model=creator,
        return_estimator=return_estimator,
        return_inspector=True,
        return_train_score=True,
        scoring=scoring,
        search_params=search_params,
        **extra_params,
    )

scores, model, inspector = out

if fold is not None:
    i_fold = fold % N_REPEATS
    i_repeat = fold // N_REPEATS
    scores["repeat"] = i_repeat
    scores["fold"] = i_fold

fname = f"{target}_{model_name}{optionals_suffix}_cv_scores{suffix}.csv"
if not IS_DEBUG_TEST:
    logger.info(f"Saving CV scores to {fname}")
    scores.to_csv(out_dir / fname, sep=";")

elapsed_time = time.time() - start_time
logger.info(
    "Elapsed CV time {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
)

logger.info("Predicting fold probabilities")
try:
    if predict_proba == "proba":
        fold_predictions = inspector.folds.predict_proba()
    elif predict_proba == "decision":
        fold_predictions = inspector.folds.decision_function()
    else:
        fold_predictions = inspector.folds.predict()
    fname = f"{target}_{model_name}" f"{optionals_suffix}_fold_predictions{suffix}.csv"
    if not IS_DEBUG_TEST:
        fold_predictions.to_csv(out_dir / fname, sep=";")
except Exception as e:  # noqa: BLE001
    logger.error(e)

if model is not None and (fold is None or fold == 0):
    logger.info("Validating model")
    y_true = validate_data[y]
    y_pred = model.predict(validate_data[X])

    data_for_df = {"y_true": y_true, "y_pred": y_pred}

    if predict_proba == "decision":
        y_probas = model.decision_function(validate_data[X])
        data_for_df["y_probas"] = y_probas
    elif predict_proba == "proba":
        y_probas = model.predict_proba(validate_data[X])[:, 1]
        data_for_df["y_probas"] = y_probas
    fname = f"{target}_{model_name}{optionals_suffix}_validation{suffix}.csv"
    val_df = pd.DataFrame(data_for_df)
    if not IS_DEBUG_TEST:
        val_df.to_csv(out_dir / fname, sep=";")
        logger.info(f"Saving validation results to {fname}")
elapsed_time = time.time() - start_time
logger.info(
    "Elapsed time {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
)
logger.info("Done!")
