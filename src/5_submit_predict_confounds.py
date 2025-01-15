import os
from pathlib import Path

cwd = os.getcwd()
log_dir_base = Path(cwd) / "logs"
log_dir_base.mkdir(exist_ok=True)

env = "julearn"

data_dir = "/data/project/ukb_rls/data/features"
exec_string = (
    f"5_predict_confounds_baseline.py --model $(model) --target $(target) "
    f" --data {data_dir}"

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
        # (t_target, "svm"),
        # (t_target, "et"),
        # (t_target, 'rf'),
        # (t_target, 'gssvm'),
        (t_target, "gssvm_rbf"),
        (t_target, 'gsrf'),
        (t_target, 'gset'),
        (t_target, 'gslinearsvm'),
        # (t_target, 'linearsvm'),
        (t_target, 'linearsvchc'),
        (t_target, 'logithc'),
        (t_target, 'stacked_linearsvcheursiticc'),
        # (t_target, 'dummy'),
        # (t_target, 'dummy_stratified'),
        # (t_target, 'optunasvm'),
        # (t_target, 'optunasvm_rbf'),
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

    with open(f"predict_all_{t_target}_confounds.submit", "w") as f:
        f.writelines(preamble)
        for target, model in to_run:
            t_log_dir = log_dir / model
            t_log_dir.mkdir(exist_ok=True, parents=True)
            log_prefix = (
                f"{t_log_dir.as_posix()}/predict_{target}_{model}{log_suffix}"
            )
            f.write(f"log            = {log_prefix}.log\n")
            f.write(f"output         = {log_prefix}.out\n")
            f.write(f"error          = {log_prefix}.err\n")
            f.write(f"target={target}\n")
            f.write(f"model={model}\n")
            f.write("queue\n\n")
