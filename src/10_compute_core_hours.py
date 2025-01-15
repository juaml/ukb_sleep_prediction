# %%
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib_htcondor.ui.treeparser import parse


# %% Select which feature set to compute core hours on
do_confounds = False

# %%
targets = [
    "pheno_extreme",
    "sleep_duration_extreme",
    "morning_extreme",
    "morning_evening_person_extreme",
    "nap_day_extreme",
    "snoring",
    "doze_day",
]
models = [
    "linearsvchc",
    "logithc",
    "stacked_linearsvcheursiticc",
    "dummy",
    "dummy_stratified",
    "gssvm_rbf",
    "gsrf",
    "gslinearsvm",
    "gset",
]

if do_confounds:
    models.remove("stacked_linearsvcheursiticc")

start_tag = "- julearn - INFO - ===== Lib Versions ====="
end_tag = "- NIMRLS - INFO - Elapsed time"
normal_tag = "- NIMRLS -"
julearn_tag = "- julearn -"

if not do_confounds:
    log_path = Path(__file__).parent / "logs"
else:
    log_path = Path(__file__).parent / "logs_confounds"
core_hours_data = {
    "target": [],
    "model": [],
    "seconds": [],
    "status": [],
}


# Go through the log files and check the start + end timestamps
for t_target in targets:
    t_log_path = log_path / t_target
    for t_model in models:
        log_file = t_log_path / t_model / f"predict_{t_target}_{t_model}.out"
        err_file = log_file.with_suffix(".err")
        print(log_file)
        start_time = None
        end_time = None
        if not log_file.exists():
            print(f"{t_target} {t_model} not found")
        else:
            with open(log_file, "r") as f:
                lines = f.readlines()
                status = "unknown"
                n_trials = -1
                for line in lines:
                    if start_tag in line:
                        print(f"{t_target} {t_model} {line}")
                        if start_time is not None:
                            raise ValueError(
                                f"Start time duplicated in {log_file}"
                            )
                        start_time = datetime.strptime(
                            line.split(start_tag)[0].strip(),
                            r"%Y-%m-%d %H:%M:%S,%f",
                        )
                        status = "running"
                    elif end_tag in line:
                        if start_time is None:
                            raise ValueError(
                                f"Start time not found in {log_file} before end time"
                            )
                        if end_time is not None:
                            raise ValueError(
                                f"End time duplicated in {log_file}"
                            )
                        end_time = datetime.strptime(
                            line.split(end_tag)[0].strip(),
                            r"%Y-%m-%d %H:%M:%S,%f",
                        )
                        status = "done"

            if start_time is None:
                raise ValueError(f"Start time not found in {log_file}")
            if end_time is None:
                print(
                    f"End time not found in {log_file}, maybe still running?"
                )
                if normal_tag in line:
                    end_time = datetime.strptime(
                        line.split(normal_tag)[0].strip(),
                        r"%Y-%m-%d %H:%M:%S,%f",
                    )
                else:
                    end_time = datetime.strptime(
                        line.split(julearn_tag)[0].strip(),
                        r"%Y-%m-%d %H:%M:%S,%f",
                    )
            core_seconds = (end_time - start_time).total_seconds()
            core_hours_data["target"].append(t_target)
            core_hours_data["model"].append(t_model)
            core_hours_data["seconds"].append(core_seconds)
            core_hours_data["status"].append(status)


# %%
if not do_confounds:
    batch_ids = {
        "gsrf": {
            "pheno_extreme": "jht-7517ab4686cd11ef81ed08c0eb24a372-l0",
            "doze_day": "jht-8077c18286ce11efade308c0eb24a8ca-l0",
            "sleep_duration_extreme": "jht-80eeab0886ce11ef8dae7cc2551e9c80-l0",
            "morning_extreme": "jht-836b574686ce11ef93077cc2551e9bd0-l0",
            "morning_evening_person_extreme": "jht-81c6c0ec86ce11efaf8a98039baf1ef0-l0",
            "nap_day_extreme": "jht-7f8dbf1086ce11ef85cd08c0eb24a372-l0",
            "snoring": "jht-7fe51af886ce11ef94d97cc2551e9c38-l0",
        },
        "gslinearsvm": {
            "pheno_extreme": "jht-fcb4dcf6868a11efac75248a071e3876-l0",
            "doze_day": "jht-fc705766868a11efa256248a071e3876-l0",
            "sleep_duration_extreme": "jht-9c260b28867911efadaf08c0eb24a372-l0",
            "morning_extreme": "jht-0156654a868b11ef8e7e7cc2551e9bd0-l0",
            "morning_evening_person_extreme": "jht-ffeb4d88868a11efb8887cc2551e9baa-l0",
            "nap_day_extreme": "jht-342c3462866911efbef908c0eb24a372-l0",
            "snoring": "jht-ddbebbda868711efbc2208c0eb24a372-l0",
        },
        "gset": {
            "pheno_extreme": "jht-bf67a4a486db11efa0bc08c0eb24a372-l0",
            "doze_day": "jht-4866d76a86e711ef856908c0eb24a372-l0",
            "sleep_duration_extreme": "jht-487d8f5a86e711efa45f08c0eb24a8ca-l0",
            "morning_extreme": "jht-4872371886e711efbf887cc2551e9c80-l0",
            "morning_evening_person_extreme": "jht-e6c1c85486ea11ef9ecd08c0eb24a372-l0",
            "nap_day_extreme": "jht-904a2e8e86eb11ef8d67b8cef65d91f6-l0",
            "snoring": "jht-bd67ee8886eb11ef96b8b8599fc3d156-l0",
        },
        "gssvm_rbf": {
            "pheno_extreme": "jht-953f704486ef11efa6aa08c0eb24a372-l0",
            "doze_day": "jht-92fb35cc8cb511efa863248a071e3876-l0",
            "sleep_duration_extreme": "jht-26689690875411ef8c0d7cc2551e9c80-l0",
            "morning_extreme": "jht-499b755e87d311efb39e08c0eb24a372-l0",
            "morning_evening_person_extreme": "jht-4f480672873b11efa40208c0eb24a8ca-l0",
            "nap_day_extreme": "jht-c94c0f4692c511ef9bff08c0eb24a372-l0",
            "snoring": "jht-92f45e1c950e11efaca07cc2551e9c38-l0",
        },
    }

    shared_joblib_dir = Path("/data/project/ukb_rls/joblib_htcondor_shared/")
else:
    batch_ids = {
        "gssvm_rbf": {
            "morning_evening_person_extreme": "jht-3d30203ccd4011ef80bc0002c94b180a-l0",
            "nap_day_extreme": "jht-5d8e0d7cb6d811efaf5d08c0eb24a372-l0",
            "doze_day": "jht-af91891e9f6d11efb8b7b8599fc3d156-l0",
        },
    }
    shared_joblib_dir = Path(
        "/data/project/ukb_rls/joblib_htcondor_shared_confounds/"
    )
# %%
for t_model, t_batchs in batch_ids.items():
    for t_target, t_id in t_batchs.items():
        print(f"Processing {t_target} {t_model} {t_id}")
        tree = parse(shared_joblib_dir / ".jht-meta" / f"{t_id}.json")
        t_batch_time = tree.get_core_hours() * 3600

        core_hours_data["target"].append(t_target)
        core_hours_data["model"].append(t_model)
        core_hours_data["seconds"].append(t_batch_time)
        core_hours_data["status"].append("done")


# %%
core_hours_df = pd.DataFrame(core_hours_data)
core_hours_df["hours"] = core_hours_df["seconds"] / 3600
print(core_hours_df.sum())
# %%

if not do_confounds:
    out_fname = Path(__file__).parent / "results" / "core_hours.csv"
else:
    out_fname = Path(__file__).parent / "results_confounds" / "core_hours.csv"
core_hours_df.to_csv(out_fname, index=False, sep=";")
# %%
