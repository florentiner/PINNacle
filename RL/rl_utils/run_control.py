"""Контроль жизненного цикла долгого запуска: дедлайн и мягкая остановка.

Одна траектория абляции идёт 1–2 часа, поэтому запуск на 1000 траекторий
никогда не завершится сам. Здесь два механизма, чтобы запуск всегда
заканчивался предсказуемо и с сохранённым результатом:

  * дедлайн по стенным часам (`--max-hours`) — тренер прекращает набор новых
    траекторий, сохраняет модель агента и делает финальную выгрузку;
  * мягкая остановка по SIGTERM/SIGINT — вместо мгновенной смерти процесс
    дописывает лог, сохраняет модель и синхронизируется с HF.

Прошлый прогон на сервере был убит снаружи на 3ч45м: в логах нет ни traceback,
ни OOM — процессы просто оборвались посреди обучения. Без обработчиков сигналов
такой конец не оставляет следов, поэтому здесь же пишется status.json с
причиной завершения.
"""
import json
import os
import signal
import time
import traceback


class RunControl:
    def __init__(self, max_seconds=None, status_path=None):
        self.max_seconds = float(max_seconds) if max_seconds else None
        self.status_path = status_path
        self.start_time = time.time()
        self.stop_requested = False
        self.stop_reason = None
        self._previous_handlers = {}

    # --- дедлайн ---

    @property
    def elapsed(self):
        return time.time() - self.start_time

    def time_left(self):
        if self.max_seconds is None:
            return None
        return self.max_seconds - self.elapsed

    def should_stop(self):
        """True, если пора завершаться (сигнал или исчерпан бюджет времени)."""
        if self.stop_requested:
            return True
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            self.stop_requested = True
            self.stop_reason = (
                f"достигнут лимит времени --max-hours "
                f"({self.max_seconds / 3600:.2f} ч)"
            )
            return True
        return False

    # --- сигналы ---

    def install_signal_handlers(self):
        def handler(signum, frame):
            name = signal.Signals(signum).name
            if self.stop_requested:
                # Второй сигнал — уходим сразу.
                print(f"\n⛔ Повторный {name}: выходим немедленно.", flush=True)
                self.write_status("killed", f"повторный {name}")
                os._exit(130)
            self.stop_requested = True
            self.stop_reason = f"получен сигнал {name}"
            print(
                f"\n⚠️  Получен {name}: доработаем текущую траекторию, сохраним "
                f"модель и выгрузим результаты. Повторный сигнал прервёт сразу.",
                flush=True,
            )

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous_handlers[sig] = signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    # --- статус запуска ---

    def write_status(self, state, detail=None, extra=None):
        """Пишет results/status.json — чтобы причина конца была видна и на HF."""
        if not self.status_path:
            return
        payload = {
            "state": state,
            "detail": detail or self.stop_reason,
            "elapsed_s": round(self.elapsed, 1),
            "max_seconds": self.max_seconds,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
        }
        if extra:
            payload.update(extra)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.status_path)), exist_ok=True)
            with open(self.status_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            traceback.print_exc()
