"""Atomic checkpoint save / auto-resume for causal runs (Kaggle 12h chaining)."""
import os
import shutil

import numpy as np
import torch


CKPT_NAME = "ckpt_latest.pt"


def save(cfg, state):
    path = os.path.join(cfg.outdir, CKPT_NAME)
    tmp = path + ".tmp"
    payload = dict(state)
    payload["torch_rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()
    payload["cfg"] = {k: v for k, v in vars(cfg).items()}
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_resume(cfg):
    """Look for a checkpoint in resume_dir (if given) or outdir. Returns state or None."""
    candidates = []
    if cfg.resume_dir:
        candidates.append(os.path.join(cfg.resume_dir, CKPT_NAME))
        # also allow pointing at the parent run dir
        for root, _, files in os.walk(cfg.resume_dir):
            if CKPT_NAME in files:
                candidates.append(os.path.join(root, CKPT_NAME))
    candidates.append(os.path.join(cfg.outdir, CKPT_NAME))
    for c in candidates:
        if os.path.exists(c):
            payload = torch.load(c, map_location="cpu", weights_only=False)
            saved_cfg = payload.pop("cfg", {})
            for key in ("case", "encoding", "causal", "windows", "n_t", "n_s", "seed"):
                if key in saved_cfg and saved_cfg[key] != getattr(cfg, key):
                    raise RuntimeError(
                        f"resume mismatch: checkpoint has {key}={saved_cfg[key]} "
                        f"but current cfg has {getattr(cfg, key)}")
            torch.set_rng_state(payload.pop("torch_rng_state").cpu())
            cu = payload.pop("cuda_rng_state", None)
            if cu is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(cu.cpu())
            # if resuming from a foreign dir (previous Kaggle session output),
            # bring its artifacts over so logs/arrays keep accumulating
            src_dir = os.path.dirname(c)
            if os.path.abspath(src_dir) != os.path.abspath(cfg.outdir):
                _import_artifacts(src_dir, cfg.outdir)
            print(f"[RESUME] from {c}: window {payload['window']} "
                  f"stage {payload['stage']} it {payload['it']}")
            return payload
    return None


def _import_artifacts(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name == CKPT_NAME or os.path.exists(d):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
