"""Поиск чекпоинта для продолжения обучения между сессиями (Kaggle/сервер).

Сессии конечны (12-часовой лимит Kaggle), поэтому обучение агента продолжается
по цепочке сессий: каждая новая начинает с чекпоинта предыдущей вместо нуля.

`resolve_resume_checkpoint` ищет в HF-датасете результатов последний запуск
пары (pde, режим абляции) и возвращает:
  * `kind="final"` — полный чекпоинт model/agent_final.pt (веса, таргеты,
    состояния Adam, steps_done);
  * `kind="snapshots"` — фоллбек: последние периодические снапшоты голов
    rl_model_snapshots/model_{optim,params}_step_N.pt (когда финальный
    чекпоинт не успел сохраниться — сессию срезал жёсткий лимит);
  * None — продолжать не с чего, обучение стартует с нуля (плюс претрен).
"""
import re


def _tag_of(path):
    parts = path.split("/")
    return parts[3] if len(parts) > 4 else None


def _snap_step(path):
    match = re.search(r"model_optim_step_(\d+)\.pt$", path)
    return int(match.group(1)) if match else -1


def resolve_resume_checkpoint(hf_repo, prefix, pde, mode, quiet=False):
    from huggingface_hub import HfApi, hf_hub_download

    base = f"{prefix}/{pde}/{mode}/"
    try:
        files = HfApi().list_repo_files(hf_repo, repo_type="dataset")
    except Exception as exc:
        print(f"⚠️ resume: не удалось прочитать {hf_repo}: {exc}")
        return None

    # теги-запуски: только начинающиеся с даты (отсекаем smoke/служебные)
    tags = sorted({
        _tag_of(f) for f in files
        if f.startswith(base) and _tag_of(f) and _tag_of(f)[:2] == "20"
    })
    if not tags and not quiet:
        print(f"resume: прошлых запусков в {hf_repo}/{base} нет — старт с нуля.")

    for tag in reversed(tags):
        troot = f"{base}{tag}/"

        final_path = f"{troot}model/agent_final.pt"
        if final_path in files:
            local = hf_hub_download(hf_repo, repo_type="dataset", filename=final_path)
            print(f"resume: найден полный чекпоинт запуска {tag}.")
            return {"kind": "final", "path": local, "tag": tag}

        snaps = [f for f in files if f.startswith(f"{troot}rl_model_snapshots/model_optim_step_")]
        if snaps:
            best = max(snaps, key=_snap_step)
            step = _snap_step(best)
            params_path = f"{troot}rl_model_snapshots/model_params_step_{step}.pt"
            if params_path in files:
                local_o = hf_hub_download(hf_repo, repo_type="dataset", filename=best)
                local_p = hf_hub_download(hf_repo, repo_type="dataset", filename=params_path)
                print(f"resume: финального чекпоинта в {tag} нет, "
                      f"берём снапшоты шага {step} (фоллбек).")
                return {"kind": "snapshots", "optim": local_o, "params": local_p,
                        "steps_done": step, "tag": tag}

    return None
