"""Выключатель задач на Kaggle. Запущенный кернел через API остановить нельзя (отмена сессии
отвечает 403, замена версии прежнюю сессию не снимает; проверено 11 октября), поэтому скрипты
сами спрашивают, не отменены ли они:

  * при старте — файл experiments/rl_arch/queue/killed.txt из клонированного репозитория
    (кернел клонирует ветку в момент старта, так что запись, запушенная пока он стоит в
    очереди Kaggle, его остановит);
  * по ходу работы — тот же список в HF: rl_arch/control/killed.txt (читается без токена),
    чтобы остановить уже идущую задачу между сидами, раундами или шагами.

Строка списка — регулярное выражение, которое ищется в теге задачи; пустые строки и строки
с # пропускаются.
"""
import os
import re
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = os.path.join(_HERE, "queue", "killed.txt")
_URL = "https://huggingface.co/datasets/danil-e/pinnacle-optuna-db/resolve/main/rl_arch/control/killed.txt"
_last = [0.0, []]


def _parse(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _patterns(remote, every):
    pats = []
    try:
        with open(_LOCAL, encoding="utf-8") as f:
            pats += _parse(f.read())
    except OSError:
        pass
    if remote:
        if time.time() - _last[0] >= every:
            _last[0] = time.time()
            try:
                req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    _last[1] = _parse(r.read().decode("utf-8", "replace"))
            except Exception:
                pass                     # файла нет или сеть недоступна: не отменено
        pats += _last[1]
    return pats


def killed(tag, remote=False, every=600):
    """Отменена ли задача с таким тегом."""
    for p in _patterns(remote, every):
        try:
            if re.search(p, tag or ""):
                return p
        except re.error:
            continue
    return None


def exit_if_killed(tag, remote=False, every=600, what="задача"):
    p = killed(tag, remote, every)
    if p:
        print(f"{what} {tag!r} отменена записью {p!r} в списке killed.txt: выход без работы", flush=True)
        sys.exit(0)
