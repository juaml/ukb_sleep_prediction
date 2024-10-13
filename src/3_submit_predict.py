import os
from pathlib import Path

cwd = os.getcwd()
log_dir_base = Path(cwd) / "logs"
log_dir_base.mkdir(exist_ok=True)

env = "julearn"

data_dir = "/data/project/ukb_rls/data/features"
exec_string = (
    f"3_predict.py --model $(model) --target $(target) --data {data_dir}"
    f" $(extra_params)"
)

log_suffix = ""

targets = [
    "pheno_extreme",
    "sleep_duration_extreme",
    "morning_extreme",
    "morning_evening_person_extreme",
    "nap_day_extreme",
    "snoring",
    "doze_day",
]

for t_target in targets:
    log_dir = log_dir_base / t_target
    log_dir.mkdir(exist_ok=True, parents=True)
    log_suffix = ""
    to_run = [
        # (t_target, "svm", None),
        # (t_target, "et", None),
        # (t_target, 'rf', None),
        # (t_target, 'gssvm', None),
        (t_target, "gssvm_rbf", None),
        (t_target, 'gsrf', None),
        (t_target, 'gset', None),
        (t_target, 'gslinearsvm', None),
        # (t_target, 'linearsvm', None),
        (t_target, 'linearsvchc', None),
        (t_target, 'logithc', None),
        (t_target, 'stacked_linearsvcheursiticc', None),
        (t_target, 'dummy', None),
        (t_target, 'dummy_stratified', None),
        # (t_target, 'optunasvm', None),
        # (t_target, 'optunasvm_rbf', None),
    ]

    preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 16G
request_disk   = 0

# Executable
initial_dir    = {cwd}
executable     = {cwd}/run_in_venv.sh
transfer_executable = False

arguments      = {env} python {exec_string}

"""

    with open(f"predict_all_{t_target}.submit", "w") as f:
        f.writelines(preamble)
        for target, model, extra_params in to_run:
            t_log_dir = log_dir / model
            t_log_dir.mkdir(exist_ok=True, parents=True)
            log_prefix = (
                f"{t_log_dir.as_posix()}/predict_{target}_{model}{log_suffix}"
            )
            f.write(f"log            = {log_prefix}.log\n")
            f.write(f"output         = {log_prefix}.out\n")
            f.write(f"error          = {log_prefix}.err\n")
            if extra_params is None:
                extra_params = ""
            
            f.write(f"target={target}\n")
            f.write(f"model={model}\n")
            f.write(f"extra_params={extra_params}\n")
            f.write("queue\n\n")
