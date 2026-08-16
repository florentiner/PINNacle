#!/usr/bin/env python
"""
Push a kernel folder requesting a specific accelerator (machine shape).

The kaggle CLI never fills ApiSaveKernelRequest.machine_shape, and Kaggle's
server default for API pushes is the P100 — whose sm_60 is unsupported by the
PyTorch build in current Kaggle images. This wrapper patches the request so
GPU kernels get NvidiaTeslaT4 (T4 x2) instead.

Usage (KAGGLE_API_TOKEN must be set):
    python _push_with_shape.py <kernel_folder> [machine_shape]
"""
import sys

folder = sys.argv[1]
shape = sys.argv[2] if len(sys.argv) > 2 else "NvidiaTeslaT4"

from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient

_orig_save = KernelsApiClient.save_kernel


def _save_with_shape(self, request):
    if shape and getattr(request, "enable_gpu", False):
        request.machine_shape = shape
        print(f"Requesting machine shape: {shape}")
    return _orig_save(self, request)


KernelsApiClient.save_kernel = _save_with_shape

# `import kaggle` authenticates once via KAGGLE_API_TOKEN (and consumes the
# env var) — reuse that instance instead of authenticating a second time.
import io
import contextlib

import kaggle

# The CLI prints "Kernel push error: ..." (quota exhausted, bad slug, ...) and
# still returns normally — a silent failure that reads as a successful launch.
# Mirror the output and turn any such line into a non-zero exit code.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    kaggle.api.kernels_push_cli(folder, None)
out = _buf.getvalue()
sys.stdout.write(out)
sys.stdout.flush()
if "push error" in out.lower() or "successfully pushed" not in out.lower():
    sys.exit(2)
