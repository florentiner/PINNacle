# Auto-generated Kaggle script kernel — PRIVATE (contains tokens; do not commit).
# Runs one Optuna study for a PINNacle PDE and syncs its SQLite DB to HF after each trial.
import json, os, subprocess, sys, time, threading

CONFIG = json.loads(r'''__CONFIG_JSON__''')

HF_REPO_ID   = CONFIG["hf_repo"]
HF_TOKEN     = CONFIG["hf_token"]
PDE          = CONFIG["pde"]
VALUE_TYPE   = CONFIG["value_type"]
N_TRIALS     = CONFIG["n_trials"]
N_PROCESSES  = CONFIG["n_processes"]
TIMEOUT_HOURS= CONFIG["timeout_hours"]
DB_FILENAME  = CONFIG["db_filename"]
STUDY_NAME   = CONFIG["study_name"]
GITHUB_REPO  = CONFIG["github_repo"]
BRANCH       = CONFIG["branch"]

print(f"=== kernel: {PDE} | value_type={VALUE_TYPE} | {N_TRIALS} trials x {N_PROCESSES} workers | study={STUDY_NAME} ===", flush=True)

# ── deps (Kaggle base image is missing optuna + friends) ────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "optuna", "sqlalchemy", "huggingface_hub", "scikit-optimize"], check=False)

# ── clone the repo (brings ns2d_liddriven_optuna.py + src/pde + optuna_trainer) ─
if not os.path.exists("PINNacle"):
    subprocess.run(["git", "clone", "-b", BRANCH, "--single-branch", GITHUB_REPO, "PINNacle"], check=True)
os.chdir("PINNacle")
sys.path.insert(0, os.getcwd())
os.environ["DDEBACKEND"] = "pytorch"
print("cwd:", os.getcwd(), flush=True)

DB_PATH = f"optuna_studies/{DB_FILENAME}"
os.makedirs("optuna_studies", exist_ok=True)

# ── HF Hub sync ────────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError

def hf_download_db():
    try:
        hf_hub_download(repo_id=HF_REPO_ID, filename=DB_PATH, repo_type="dataset",
                        token=HF_TOKEN, local_dir=".")
        return os.path.exists(DB_PATH)
    except EntryNotFoundError:
        return False
    except Exception as e:
        print("  HF download:", e, flush=True)
        return False

_db_lock = threading.Lock()
def hf_upload_db():
    if not HF_TOKEN or not os.path.exists(DB_PATH):
        return
    with _db_lock:
        try:
            upload_file(path_or_fileobj=DB_PATH, path_in_repo=DB_PATH, repo_id=HF_REPO_ID,
                        repo_type="dataset", token=HF_TOKEN,
                        commit_message=f"update {DB_FILENAME}")
            print(f"  DB synced to HF ({os.path.getsize(DB_PATH)//1024} KB)", flush=True)
        except Exception as e:
            print("  HF upload:", e, flush=True)

if hf_download_db():
    print(f"  Resuming study from HF DB ({os.path.getsize(DB_PATH)//1024} KB)", flush=True)
else:
    print("  No DB on Hub — starting fresh", flush=True)

# ── launch workers (each runs the Optuna study against a shared local SQLite DB) ─
SCRIPT = f"experiments/optuna_multi_pde/{PDE}_optuna.py"
cmd = [sys.executable, SCRIPT,
       "--db-path", DB_PATH,
       "--study-name", STUDY_NAME,
       "--n-trials", str(N_TRIALS),
       "--timeout-hours", str(TIMEOUT_HOURS),
       "--sampler", "tpe",
       "--value-type", VALUE_TYPE]

import torch
compat = [i for i in range(torch.cuda.device_count())
          if (lambda p: p.major * 10 + p.minor >= 70)(torch.cuda.get_device_properties(i))]

procs = []
for i in range(N_PROCESSES):
    env = os.environ.copy()
    env["DDEBACKEND"] = "pytorch"
    env["CUDA_VISIBLE_DEVICES"] = str(compat[i % len(compat)]) if compat else ""
    procs.append(subprocess.Popen(cmd, env=env))
    print(f"Worker {i} started (PID {procs[-1].pid})", flush=True)

# ── background thread: upload DB whenever the completed-trial count grows ───────
import optuna as _opt
def sync_loop():
    storage = f"sqlite:///{os.path.abspath(DB_PATH)}"
    last = 0
    while any(p.poll() is None for p in procs):
        time.sleep(15)
        if not os.path.exists(DB_PATH):
            continue
        try:
            s = _opt.load_study(study_name=STUDY_NAME, storage=storage)
            n = sum(1 for t in s.trials if t.state in (
                _opt.trial.TrialState.COMPLETE, _opt.trial.TrialState.FAIL, _opt.trial.TrialState.PRUNED))
            if n > last:
                print(f"  Trial {n} done → syncing", flush=True)
                hf_upload_db()
                last = n
        except Exception:
            pass

t = threading.Thread(target=sync_loop, daemon=True)
t.start()
for p in procs:
    p.wait()
    print(f"Worker PID {p.pid} exit {p.returncode}", flush=True)
t.join(timeout=30)
hf_upload_db()
print("All workers done. DB synced to HF.", flush=True)
