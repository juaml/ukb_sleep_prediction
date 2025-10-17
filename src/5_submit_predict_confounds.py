import os
from pathlib import Path

cwd = os.getcwd()
log_dir_base = Path(cwd) / "logs"
log_dir_base.mkdir(exist_ok=True)

env = "julearn"

data_dir = "/data/project/ukb_rls/data/features"
exec_string = (
    f"5_predict_confounds_baseline.py --model $(model) --target $(target) "
    f" --data {data_dir} --confounds $(confounds)"
)

log_suffix = ""

confounds = ["full", "reduced"]


# Run only best models

to_run = [
    # ("doze_day", "logithc"),  # DONE
    # ("morning_evening_person_extreme", "logithc"),  # DONE
    # ("morning_extreme", "gslinearsvm"),  # DONE
    # ("nap_day_extreme", "logithc"),  # DONE
    # ("pheno_extreme","logithc" ),  # DONE
    # ("sleep_duration_extreme", "logithc"),  # DONE
    # ("snoring", "logithc"),  # DONE

    ("doze_day", "gssvm_rbf"),  # LONG DONE
    # ("morning_evening_person_extreme", "gslinearsvm"),  # DONE
    # ("morning_extreme", "gslinearsvm"),  # Repeated
    ("nap_day_extreme", "gssvm_rbf"),  # Full DONE
    # ("pheno_extreme", "gslinearsvm"),  # DONE
    # ("sleep_duration_extreme", "gslinearsvm"),  # DONE
    # ("snoring", "gslinearsvm"),  # DONE

    # ("doze_day", "gssvm_rbf"), # Repeated
    # ("morning_evening_person_extreme","gssvm_rbf" ),  # LONG DONE
    # ("morning_extreme", "gslinearsvm"),  # Repeated
    # ("nap_day_extreme", "gssvm_rbf"),  # Repeated
    # ("pheno_extreme", "linearsvchc"),  # DONE
    # ("sleep_duration_extreme", "gset"),  # DONE
    # ("snoring", "gslinearsvm"),  # Repeated

    # ("doze_day", "logithc"),  # Repeated
    # ("morning_evening_person_extreme", "logithc"), # Repeated
    # ("morning_extreme", "gslinearsvm"),  # Repeated
    # ("nap_day_extreme", "logithc"),  # Repeated
    # ("pheno_extreme", "logithc"),  # Repeated
    # ("sleep_duration_extreme", "logithc"),  # Repeated
    # ("snoring", "logithc"),  # Repeated
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

with open(f"predict_all_confounds.submit", "w") as f:
    f.writelines(preamble)
    for t_target, model in to_run:
        t_log_dir = log_dir_base / t_target / model
        t_log_dir.mkdir(exist_ok=True, parents=True)
        for confound_set in confounds:
            log_prefix = (
                f"{t_log_dir.as_posix()}/"
                f"predict_{t_target}_{model}_{confound_set}_{log_suffix}"
            )
            f.write(f"log            = {log_prefix}.log\n")
            f.write(f"output         = {log_prefix}.out\n")
            f.write(f"error          = {log_prefix}.err\n")
            f.write(f"target={t_target}\n")
            f.write(f"confounds={confound_set}\n")
            f.write(f"model={model}\n")
            f.write("queue\n\n")
