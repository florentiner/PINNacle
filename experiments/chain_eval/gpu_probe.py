#!/usr/bin/env python
"""Минимальная проба: есть ли на аккаунте реальный GPU и рабочий torch."""
import os, subprocess, sys
subprocess.run(["nvidia-smi"], check=False)
try:
    import torch
    ok = torch.cuda.is_available()
    print(f"PROBE torch={torch.__version__} cuda_available={ok}", flush=True)
    if ok:
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).sum().item()
        print(f"PROBE devices={n} {names} matmul_ok={abs(y) < float('inf')}", flush=True)
        print("PROBE RESULT: GPU OK", flush=True)
    else:
        print("PROBE RESULT: НЕТ GPU", flush=True)
except Exception as e:
    print(f"PROBE RESULT: ОШИБКА {type(e).__name__}: {e}", flush=True)
