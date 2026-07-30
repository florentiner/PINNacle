"""Построчный CSV-лог метрик по завершённым траекториям.

Формат строки (одна траектория = одна строка):

    run_timestamp,pde_name,value_type,smoke_test,chain_key,seed,
    mse_op,mse_bnd,mse_total,l2re_op,l2re_bnd,l2re,l2re_total,elapsed_s,chain_json

Соотношения между колонками:
    mse_total  = mse_op + mse_bnd
    l2re       = sqrt(l2re_op^2 + l2re_bnd^2)
    l2re_total = l2re_op + l2re_bnd

`chain_json` — цепочка действий агента за траекторию:
    [{"optimizer": "Adam", "lr": 0.001, "epochs": 1000}, ...]
`chain_key` — та же цепочка как ключ: имена оптимизаторов в нижнем регистре
через "_", подряд идущие повторы схлопнуты (adam,adam,lbfgs -> adam_lbfgs).
"""
import csv
import json
import math
import os

# Обязательные колонки в согласованном порядке — их формат менять нельзя.
CSV_FIELDS_REQUIRED = [
    "run_timestamp",
    "pde_name",
    "value_type",
    "smoke_test",
    "chain_key",
    "seed",
    "mse_op",
    "mse_bnd",
    "mse_total",
    "l2re_op",
    "l2re_bnd",
    "l2re",
    "l2re_total",
    "elapsed_s",
    "chain_json",
]

# Дополнительные колонки — ошибка обучения САМОГО агента и состояние эпизода.
# Дописаны в конец, чтобы первые 15 колонок остались ровно те же и парсеры,
# читающие по позиции, не сломались.
#   ablation           — какой компонент DQN-стека выключен
#   agent_loss_optim   — средний Huber-лосс головы выбора оптимизатора (TD-ошибка)
#   agent_loss_param   — средний лосс голов гиперпараметров
#   agent_td_abs       — средний |y - Q(s,a)| (сырая TD-ошибка)
#   agent_q_abs        — средний |Q(s,a)|, для контроля масштаба
#   agent_tr_drop_frac — доля батча, выкинутая trust-region маской
CSV_FIELDS_EXTRA = [
    "ablation",
    "done",
    "steps",
    "final_loss",
    "total_reward",
    "agent_loss_optim",
    "agent_loss_param",
    "agent_td_abs",
    "agent_q_abs",
    "agent_tr_drop_frac",
]

CSV_FIELDS = CSV_FIELDS_REQUIRED + CSV_FIELDS_EXTRA


def build_chain_entries(actions):
    """actions: список action-словарей агента -> список записей для chain_json."""
    entries = []
    for action in actions:
        params = action.get("params", {}) or {}
        entry = {"optimizer": action.get("type")}
        if "lr" in params:
            entry["lr"] = params["lr"]
        entry["epochs"] = action.get("epochs")
        for key, value in params.items():
            if key != "lr":
                entry[key] = value
        entries.append(entry)
    return entries


def build_chain_key(chain_entries):
    """Схлопывает подряд идущие повторы: Adam,Adam,LBFGS -> adam_lbfgs."""
    names = []
    for entry in chain_entries:
        name = str(entry.get("optimizer", "")).lower()
        if not name:
            continue
        if not names or names[-1] != name:
            names.append(name)
    return "_".join(names)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


class TrajectoryMetricsLogger:
    """Пишет CSV-строку на каждую завершённую траекторию.

    Файл лежит внутри run_dir, поэтому уезжает на HF вместе с остальными
    результатами. Пишется и flush-ится сразу, чтобы прерванный запуск не терял
    уже посчитанные траектории.
    """

    def __init__(self, csv_path, run_timestamp, pde_name, value_type,
                 seed, smoke_test=False, experiment=None):
        self.csv_path = os.path.abspath(csv_path)
        self.run_timestamp = run_timestamp
        self.pde_name = pde_name
        self.value_type = value_type
        self.seed = seed
        self.smoke_test = bool(smoke_test)
        self.experiment = experiment
        self.rows_written = 0

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def log_trajectory(self, mse_op, mse_bnd, l2re_op, l2re_bnd, elapsed_s,
                       actions, trajectory_index=None, extra=None):
        """extra — значения дополнительных колонок (см. CSV_FIELDS_EXTRA)."""
        mse_op = _finite(mse_op)
        mse_bnd = _finite(mse_bnd)
        l2re_op = _finite(l2re_op)
        l2re_bnd = _finite(l2re_bnd)

        chain_entries = build_chain_entries(actions)
        row = {
            "run_timestamp": self.run_timestamp,
            "pde_name": self.pde_name,
            "value_type": self.value_type,
            "smoke_test": self.smoke_test,
            "chain_key": build_chain_key(chain_entries),
            "seed": self.seed,
            "mse_op": mse_op,
            "mse_bnd": mse_bnd,
            "mse_total": mse_op + mse_bnd,
            "l2re_op": l2re_op,
            "l2re_bnd": l2re_bnd,
            "l2re": math.sqrt(l2re_op ** 2 + l2re_bnd ** 2),
            "l2re_total": l2re_op + l2re_bnd,
            "elapsed_s": round(float(elapsed_s), 1),
            "chain_json": json.dumps(chain_entries, ensure_ascii=False),
        }
        extra = extra or {}
        for field in CSV_FIELDS_EXTRA:
            row[field] = extra.get(field, "")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
            f.flush()
        self.rows_written += 1

        print(
            f"\n📈 Траектория записана в {os.path.basename(self.csv_path)}: "
            f"chain={row['chain_key']}, l2re={row['l2re']:.6g}, "
            f"mse_total={row['mse_total']:.6g}, elapsed={row['elapsed_s']}s, "
            f"agent_loss_optim={row.get('agent_loss_optim')}"
        )

        # Дублируем в общий поток метрик (Comet или HF), чтобы было видно онлайн.
        if self.experiment is not None:
            metrics = {
                f"trajectory/{k}": v for k, v in row.items()
                if k not in ("chain_json", "chain_key", "run_timestamp",
                             "pde_name", "value_type", "smoke_test", "ablation")
                and v != ""
            }
            metrics["trajectory/chain_len"] = len(chain_entries)
            try:
                self.experiment.log_metrics(metrics, step=trajectory_index)
            except Exception as exc:
                print(f"⚠️ Не удалось залогировать метрики траектории: {exc}")

        return row
