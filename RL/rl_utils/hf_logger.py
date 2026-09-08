"""Логгер результатов запуска в HuggingFace-датасет (замена Comet).

`HFExperiment` повторяет ту часть интерфейса comet-эксперимента, которой
пользуется код агента (`log_metrics`, `log_metric`, `log_parameters`,
`log_parameter`, `log_asset`, `log_other`, `end`), поэтому подставляется в
`rl_agent_params["exp"]` без изменений в rl_trainer/rl_algorithms.

Всё пишется локально в `run_dir`, а затем пачками синхронизируется на HF —
загружать каждый ассет отдельным коммитом слишком дорого. Раскладка запуска
разнесена по папкам, чтобы логи, результаты и модель не смешивались:

    run_dir/
        logs/log.txt                     # stdout+stderr запуска
        results/params.json              # log_parameters / log_parameter
        results/others.json              # log_other
        results/metrics.jsonl            # по строке на вызов log_metrics
        results/trajectory_metrics.csv   # строка на завершённую траекторию
        model/agent_final.pt             # обученный агент (в конце обучения)
        rl_model_snapshots/              # промежуточные снапшоты (ротация)
        assets/                          # прочие ассеты (приоритеты буфера)

В датасете всё это ложится в `<repo_path>/`, например
`runs/poisson_boltzmann_2d/no_per/2026-07-30_12-00-00_gpu01/`.

Требуется HF_TOKEN с правом записи в целевой датасет.
"""
import json
import os
import random
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


def hf_retry(fn, *args, what="HF-запрос", attempts=8, base_wait=30.0, **kwargs):
    """Повтор HF-вызова при 429 (лимит запросов) и сетевых сбоях.

    У аккаунта HF общий лимит ~1000 API-запросов на 5 минут, и он ОБЩИЙ для
    всех параллельных сессий кампании: одна `snapshot_download` буфера — это
    запрос на каждый файл (у heatinv их 212), поэтому два десятка сессий,
    стартовавших разом, выбивают лимит за минуты. Ошибка приходит как 429 с
    заголовком "Retry after N seconds" — ждём именно столько, если сказано.
    """
    import random as _random
    import re as _re

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            text = str(exc)
            is_429 = "429" in text or "rate limit" in text.lower()
            is_net = any(s in type(exc).__name__.lower() for s in ("timeout", "connection"))
            if not (is_429 or is_net) or attempt == attempts:
                raise
            wait = base_wait * (2 ** (attempt - 1))
            match = _re.search(r"[Rr]etry after (\d+)", text)
            if match:
                wait = float(match.group(1)) + 5.0
            wait = min(wait, 600.0) + _random.uniform(0, 15)
            print(f"⏳ {what}: попытка {attempt}/{attempts} не прошла "
                  f"({'лимит запросов HF' if is_429 else type(exc).__name__}); "
                  f"ждём {wait:.0f} с", flush=True)
            time.sleep(wait)



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
        keep_last_assets=5,
    ):
        self.repo_id = repo_id
        self.repo_path = repo_path.strip("/")
        self.run_dir = os.path.abspath(run_dir)
        self.assets_dir = os.path.join(self.run_dir, "assets")
        self.results_dir = os.path.join(self.run_dir, "results")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.model_dir = os.path.join(self.run_dir, "model")
        self.sync_every_sec = float(sync_every_sec)
        self.strip_solver_models = strip_solver_models
        self.keep_last_assets = keep_last_assets

        for path in (self.assets_dir, self.results_dir, self.logs_dir, self.model_dir):
            os.makedirs(path, exist_ok=True)

        self.params = {}
        self.others = {}
        self._metrics_path = os.path.join(self.results_dir, "metrics.jsonl")
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
        hf_retry(self._api.create_repo, repo_id, repo_type="dataset", private=private,
                 exist_ok=True, what="create_repo")
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
            return

        self._rotate_assets(name)

    def _rotate_assets(self, name):
        """Оставляет последние N ассетов одного семейства (например priority_step_*).

        Такие ассеты пишутся на каждом шаге обучения; без ротации датасет
        распухает, а каждая синхронизация тащит всю историю.
        """
        if not self.keep_last_assets or self.keep_last_assets <= 0:
            return

        base = os.path.basename(name)
        family = "".join(ch for ch in base if not ch.isdigit())
        folder = os.path.dirname(os.path.join(self.assets_dir, name))
        try:
            siblings = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if "".join(ch for ch in f if not ch.isdigit()) == family
            ]
        except OSError:
            return

        siblings.sort(key=lambda p: os.path.getmtime(p))
        for stale in siblings[:-self.keep_last_assets]:
            try:
                os.remove(stale)
            except OSError:
                pass

    def end(self):
        """Финальная синхронизация — вызывать в конце запуска."""
        self._sync(force=True)

    # --- внутреннее ---

    def _write_json(self, name, payload):
        with open(os.path.join(self.results_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _maybe_sync(self):
        if time.time() - self._last_sync >= self.sync_every_sec:
            self._sync()

    # Финальная выгрузка — единственная копия agent_final.pt и последних строк
    # CSV. Десятки параллельных сессий коммитят в один датасет, и HF отвечает
    # 429/409 на гонки коммитов, поэтому force-синхронизация повторяется с
    # растущей паузой, а не сдаётся с первой попытки.
    FINAL_SYNC_ATTEMPTS = 5
    FINAL_SYNC_BACKOFF_SEC = 30.0

    def _sync(self, force=False):
        self._last_sync = time.time()
        attempts = self.FINAL_SYNC_ATTEMPTS if force else 1
        for attempt in range(1, attempts + 1):
            try:
                sys.stdout.flush()
                self._api.upload_folder(
                    folder_path=self.run_dir,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    path_in_repo=self.repo_path,
                    commit_message=f"sync run {self.repo_path} (#{self._sync_count + 1})",
                    # Зеркалим ротацию: снапшоты/ассеты, удалённые локально,
                    # удаляются и на HF (паттерны ограничены папкой этого запуска).
                    delete_patterns=[
                        f"{self.repo_path}/rl_model_snapshots/*",
                        f"{self.repo_path}/assets/*",
                    ],
                )
                self._sync_count += 1
                print(f"📤 HF sync #{self._sync_count}: {self.repo_id}/{self.repo_path}")
                return
            except Exception as exc:
                # Обрыв сети не должен ронять многочасовое обучение.
                self._failed_syncs += 1
                print(f"⚠️ HF sync не удался ({self._failed_syncs}, попытка {attempt}/{attempts}): {exc}")
                if attempt < attempts:
                    pause = self.FINAL_SYNC_BACKOFF_SEC * (2 ** (attempt - 1))
                    pause += random.uniform(0, pause / 2)  # джиттер против синхронных ретраев
                    print(f"   повтор через {pause:.0f} с", flush=True)
                    time.sleep(pause)
                elif force:
                    traceback.print_exc()
                    print(f"❌ Финальная выгрузка на HF не удалась после {attempts} попыток; "
                          f"результаты остались локально в {self.run_dir}", flush=True)


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
