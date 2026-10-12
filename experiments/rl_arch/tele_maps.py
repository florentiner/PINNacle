#!/usr/bin/env python
"""
Режим состояния fulltele: карты ландшафта И телеметрия обучения в одном входе сети
(сигналы состояния обучения рядом с наблюдением, как в AOS-R 2608.01997).

Раскладка каналов (одна и та же в online_train_env.py и online_eval_env.py — обе стороны
собирают состояние функциями этого модуля, контекст приклеивают своим add_scalar_ctx):

    0..3    карты: loss_total, loss_oper, loss_bnd, delta (online_eval_env.build_state)
    4..13   телеметрия последнего действия постоянными каналами (offline_rl.tele_state, TELE_NAMES)
    14      признак «телеметрия есть»: 1 после действия с конечными лоссами, иначе 0
    15..19  скалярный контекст (--scalar-ctx), дальше описатель задачи (--pde-ctx)

Офлайновые буферы телеметрии не содержат (в дампах нет кривой лосса, градиента и весов).
Решение для RLPD: офлайновым переходам телеметрия ставится нулями, признак — 0; онлайновые
переходы несут настоящую телеметрию и признак 1. Сеть различает «телеметрии нет» и
«телеметрия нулевая», а половина батча из буфера продолжает учить ветвь карт. Старт цепочки
(нулевое состояние) совпадает с офлайновым видом: карты, телеметрия и признак нулевые.

Тёплый старт от агента на картах (4 + контекст каналов): первая свёртка расширяется, веса
новых каналов (телеметрия и признак) нулевые — сеть в момент старта считает ровно те же Q,
что исходный агент, а телеметрию подключает градиентом (widen_state_dict, widen_norm).

Нормировка: телеметрия уже закодирована в [-3, 3] фиксированными формулами, признак — 0/1;
этим каналам ставятся нулевое среднее и единичный разброс (fix_norm), иначе статистики
офлайнового буфера (все нули, разброс 1e-6) раздули бы онлайновые значения в миллион раз.
"""
from __future__ import annotations

import numpy as np

from offline_rl import TELE_CH, TELE_NAMES, state_channels

FULLTELE = "fulltele"
MAP_CH = 4
FLAG_CH = 1
FULLTELE_CH = MAP_CH + TELE_CH + FLAG_CH        # 15 каналов наблюдения до контекста
TELE_SLICE = slice(MAP_CH, MAP_CH + TELE_CH)
FLAG_IDX = MAP_CH + TELE_CH
MAP_NAMES = ("loss_total", "loss_oper", "loss_bnd", "delta")
CTX_NAMES = ("ctx_step", "ctx_budget", "ctx_prev_opt", "ctx_prev_epochs", "ctx_err")


def obs_channels(mode: str) -> int:
    """Число каналов наблюдения до контекста: 15 у fulltele, иначе как offline_rl.state_channels
    (10 у tele, 4 у остальных режимов)."""
    return FULLTELE_CH if mode == FULLTELE else state_channels(mode)


def wants_tele(mode: str) -> bool:
    """Нужна ли телеметрия действия (кривая лосса, веса до действия): tele и fulltele."""
    return mode in ("tele", FULLTELE)


def channel_names(mode: str, scalar_ctx: bool = False, n_desc: int = 0):
    """Имена каналов входа сети по порядку — для meta чекпоинта и проверок раскладки."""
    if mode == FULLTELE:
        names = list(MAP_NAMES) + ["tele_" + n for n in TELE_NAMES] + ["tele_valid"]
    elif mode == "tele":
        names = ["tele_" + n for n in TELE_NAMES]
    else:
        names = list(MAP_NAMES)
    if scalar_ctx:
        names += list(CTX_NAMES)
    return names + [f"pde_desc{i}" for i in range(int(n_desc))]


def fulltele_obs(maps, tele=None, valid=None):
    """Наблюдение fulltele (15, H, W): карты, телеметрия, признак её наличия.
    maps — (4, H, W); tele — (TELE_CH, H, W) или None (телеметрии нет: нули и признак 0);
    valid — признак явно (по умолчанию: телеметрия передана)."""
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] != MAP_CH:
        raise ValueError(f"fulltele: ждутся карты ({MAP_CH}, H, W), пришло {maps.shape}")
    h, w = maps.shape[1:]
    if tele is None:
        tele_p = np.zeros((TELE_CH, h, w), dtype=np.float32)
        ok = False if valid is None else bool(valid)
    else:
        tele_p = np.asarray(tele, dtype=np.float32)
        if tele_p.shape != (TELE_CH, h, w):
            raise ValueError(f"fulltele: ждётся телеметрия ({TELE_CH}, {h}, {w}), пришло {tele_p.shape}")
        ok = True if valid is None else bool(valid)
    flag = np.full((FLAG_CH, h, w), 1.0 if ok else 0.0, dtype=np.float32)
    return np.concatenate([maps, tele_p, flag], axis=0)


def fulltele_batch(S):
    """Офлайновые состояния (N, 4, H, W) -> (N, 15, H, W): телеметрии в буферах нет, её каналы
    и признак нулевые. Вызывается ДО приклейки контекста."""
    S = np.asarray(S, dtype=np.float32)
    if S.ndim != 4 or S.shape[1] != MAP_CH:
        raise ValueError(f"fulltele: ждутся карты (N, {MAP_CH}, H, W), пришло {S.shape}")
    n, _, h, w = S.shape
    return np.concatenate([S, np.zeros((n, TELE_CH + FLAG_CH, h, w), dtype=np.float32)], axis=1)


def fix_norm(mean, std):
    """Каналам телеметрии и признака — нулевое среднее и единичный разброс (копии массивов).
    mean/std формы (1, C, 1, 1), C >= 15; карты и контекст не трогаются."""
    mean = np.array(mean, dtype=np.float32, copy=True)
    std = np.array(std, dtype=np.float32, copy=True)
    if mean.ndim != 4 or mean.shape[1] < FULLTELE_CH:
        raise ValueError(f"fulltele: статистики нормировки формы {mean.shape}, нужно не меньше "
                         f"{FULLTELE_CH} каналов")
    mean[:, MAP_CH:FULLTELE_CH] = 0.0
    std[:, MAP_CH:FULLTELE_CH] = 1.0
    return mean, std


def first_conv_key(sd):
    """Имя весов первой свёртки сети — как его находит offline_rl.ckpt_arch (первый
    четырёхмерный тензор с ядром не больше 7)."""
    for k, v in sd.items():
        if hasattr(v, "dim") and v.dim() == 4 and v.shape[2] <= 7:
            return k
    raise ValueError("в чекпоинте нет свёрточного входа: расширение каналов не определено")


def widen_state_dict(sd_old, sd_new, old_in: int, new_in: int):
    """Веса агента на картах (old_in = 4 + контекст) -> сеть fulltele (new_in = 15 + контекст).
    Все тензоры переносятся как есть, кроме первой свёртки: её каналы карт и контекста
    ложатся на новые места, каналы телеметрии и признака получают нулевые веса. Каналы,
    которые сеть добавляет сама перед входом (координаты resnetpos), остаются первыми.
    sd_new — state_dict уже построенной широкой сети (образец форм). Возвращает новый dict."""
    import torch
    n_ctx = int(old_in) - MAP_CH
    if n_ctx < 0 or int(new_in) != FULLTELE_CH + n_ctx:
        raise ValueError(f"расширение до fulltele: у чекпоинта {old_in} каналов, у сети {new_in}; "
                         f"ждётся {MAP_CH}+k -> {FULLTELE_CH}+k с одним и тем же контекстом k")
    key = first_conv_key(sd_old)
    if set(sd_old) != set(sd_new):
        diff = sorted(set(sd_old) ^ set(sd_new))
        raise ValueError(f"расширение до fulltele: наборы параметров не совпали: {diff[:6]}")
    out = {}
    for k, v in sd_old.items():
        tgt = sd_new[k]
        if k != key:
            if tuple(v.shape) != tuple(tgt.shape):
                raise ValueError(f"расширение до fulltele: {k}: форма {tuple(v.shape)} в чекпоинте, "
                                 f"{tuple(tgt.shape)} у сети")
            out[k] = v.clone()
            continue
        lead = int(v.shape[1]) - int(old_in)             # каналы, добавленные самой сетью
        if lead < 0 or int(tgt.shape[1]) != lead + int(new_in) or tuple(v.shape[2:]) != tuple(tgt.shape[2:]) \
                or v.shape[0] != tgt.shape[0]:
            raise ValueError(f"расширение до fulltele: {k}: форма {tuple(v.shape)} в чекпоинте, "
                             f"{tuple(tgt.shape)} у сети")
        # на устройстве чекпоинта (CPU): сеть может жить на GPU, load_state_dict перенесёт сам
        w = torch.zeros(tuple(tgt.shape), dtype=v.dtype, device=v.device)
        w[:, :lead + MAP_CH] = v[:, :lead + MAP_CH]                      # координаты и карты
        if n_ctx:
            w[:, lead + FULLTELE_CH:] = v[:, lead + MAP_CH:]             # контекст — в хвост
        out[k] = w
    return out


def widen_norm(mean, std, old_in: int):
    """Статистики нормировки агента на картах -> fulltele: карты и контекст сохраняют свои
    значения, каналы телеметрии и признака получают 0 и 1."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (1, int(old_in), 1, 1) or std.shape != mean.shape:
        raise ValueError(f"расширение до fulltele: статистики формы {mean.shape}/{std.shape}, "
                         f"ждётся (1, {old_in}, 1, 1)")
    zeros = np.zeros((1, TELE_CH + FLAG_CH, 1, 1), dtype=np.float32)
    return (np.concatenate([mean[:, :MAP_CH], zeros, mean[:, MAP_CH:]], axis=1),
            np.concatenate([std[:, :MAP_CH], zeros + 1.0, std[:, MAP_CH:]], axis=1))
