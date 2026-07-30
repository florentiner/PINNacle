"""Логгер результатов запуска в HuggingFace-датасет (замена Comet).

`HFExperiment` повторяет ту часть интерфейса comet-эксперимента, которой
пользуется код агента (`log_metrics`, `log_metric`, `log_parameters`,
`log_parameter`, `log_asset`, `log_other`, `end`), поэтому подставляется в
`rl_agent_params["exp"]` без изменений в rl_trainer/rl_algorithms.

Всё пишется локально в `run_dir`, а затем пачками синхронизируется на HF —
загружать каждый ассет отдельным коммитом слишком дорого:

    run_dir/
        params.json        # log_parameters / log_parameter
        others.json        # log_other
        metrics.jsonl      # по строке на вызов log_metrics/log_metric
        log.txt            # stdout+stderr запуска (если включён Tee)
        assets/            # снапшоты агента, транзишены и пр.

В репозитории датасета всё это ложится в `<repo_path>/`, например
`runs/poisson_boltzmann_2d/no_per/2026-07-30_12-00-00_gpu01/`.

Требуется HF_TOKEN с правом записи в целевой датасет.
"""
import json
import os
import shutil
import sys
import time
import traceback

import torch


def _json_safe(value):
    """Приводит значение к JSON-сериализуемому виду (тензоры, np-скаляры и пр.)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.numel() > 1 else value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    for attr in ("item", "tolist"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:
                pass
    return str(value)


class HFExperiment:
    """Пишет метрики/параметры/ассеты локально и синхронизирует их на HF."""

    # Код агента отличает Comet-эксперимент по этому флагу (см. rl_algorithms).
    is_comet = False

    def __init__(
        self,
        repo_id,
        repo_path,
        run_dir,
        token=None,
        sync_every_sec=900,
        strip_solver_models=True,
        private=False,
    ):
        self.repo_id = repo_id
        self.repo_path = repo_path.strip("/")
        self.run_dir = os.path.abspath(run_dir)
        self.assets_dir = os.path.join(self.run_dir, "assets")
        self.sync_every_sec = float(sync_every_sec)
        self.strip_solver_models = strip_solver_models

        os.makedirs(self.assets_dir, exist_ok=True)

        self.params = {}
        self.others = {}
        self._metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._last_sync = time.time()
        self._sync_count = 0
        self._failed_syncs = 0

        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Для логирования на HF нужен HF_TOKEN с правом записи "
                f"в {repo_id} (или запускайте с --no-comet без --hf-results)."
            )

        from huggingface_hub import HfApi

        self._api = HfApi(token=token)
        self._api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"📤 HF-логгер: {repo_id}/{self.repo_path} (локально: {self.run_dir})")

    # --- интерфейс, совместимый с comet-экспериментом ---

    def log_parameters(self, params, **kwargs):
        for key, value in dict(params).items():
            self.params[str(key)] = _json_safe(value)
        self._write_json("params.json", self.params)

    def log_parameter(self, name, value, **kwargs):
        self.log_parameters({name: value})

    def log_metrics(self, metrics, step=None, **kwargs):
        record = {"step": step, "time": time.time()}
        record.update({str(k): _json_safe(v) for k, v in dict(metrics).items()})
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._maybe_sync()

    def log_metric(self, name, value, step=None, **kwargs):
        self.log_metrics({name: value}, step=step)

    def log_other(self, key, value, **kwargs):
        self.others[str(key)] = _json_safe(value)
        self._write_json("others.json", self.others)

    def log_asset(self, file_path, file_name=None, step=None, overwrite=True, **kwargs):
        """Копирует ассет в run_dir/assets; для .pt срезает solver_models.

        solver_models (снимки весов PINN) занимают ~98% веса транзишена и
        буфером не используются — см. проверку в истории задачи.
        """
        name = file_name or os.path.basename(file_path)

        # Файлы, которые и так лежат внутри run_dir (снапшоты агента, графики),
        # уедут на HF при синхронизации папки — копировать их в assets/ значит
        # залить те же данные дважды.
        src_abs = os.path.abspath(file_path)
        if os.path.commonpath([src_abs, self.run_dir]) == self.run_dir:
            return

        dst = os.path.join(self.assets_dir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        try:
            if self.strip_solver_models and name.endswith(".pt"):
                data = torch.load(file_path, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and "solver_models" in data:
                    data = {k: v for k, v in data.items() if k != "solver_models"}
                    torch.save(data, dst)
                else:
                    shutil.copyfile(file_path, dst)
            else:
                shutil.copyfile(file_path, dst)
        except Exception as exc:
            print(f"⚠️ HF-логгер: не удалось сохранить ассет {name}: {exc}")

    def end(self):
        """Финальная синхронизация — вызывать в конце запуска."""
        self._sync(force=True)

    # --- внутреннее ---

    def _write_json(self, name, payload):
        with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _maybe_sync(self):
        if time.time() - self._last_sync >= self.sync_every_sec:
            self._sync()

    def _sync(self, force=False):
        self._last_sync = time.time()
        try:
            sys.stdout.flush()
            self._api.upload_folder(
                folder_path=self.run_dir,
                repo_id=self.repo_id,
                repo_type="dataset",
                path_in_repo=self.repo_path,
                commit_message=f"sync run {self.repo_path} (#{self._sync_count + 1})",
            )
            self._sync_count += 1
            print(f"📤 HF sync #{self._sync_count}: {self.repo_id}/{self.repo_path}")
        except Exception as exc:
            # Обрыв сети не должен ронять многочасовое обучение.
            self._failed_syncs += 1
            print(f"⚠️ HF sync не удался ({self._failed_syncs}): {exc}")
            if force:
                traceback.print_exc()


class Tee:
    """Дублирует stdout/stderr в файл, чтобы лог запуска уехал на HF."""

    def __init__(self, path, stream):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        try:
            self.file.write(data)
        except Exception:
            pass

    def flush(self):
        self.stream.flush()
        try:
            self.file.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def tee_stdout(log_path):
    """Включает дублирование stdout/stderr в log_path."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    sys.stdout = Tee(log_path, sys.__stdout__)
    sys.stderr = Tee(log_path, sys.__stderr__)
    return log_path
