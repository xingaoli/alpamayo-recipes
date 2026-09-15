# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from collections import defaultdict

from rich.console import Console
from rich.pretty import pprint
import torch
import random
import numpy as np


def pformat(obj: Any) -> str:
    """Pretty format an object."""
    console = Console()
    with console.capture() as capture:
        pprint(obj, console=console, expand_all=True)
    return capture.get()


def get_param_count(nn_model: torch.nn.Module, depth: int = 2) -> dict:
    """Get parameter counts for each module."""
    if depth < 1:
        raise ValueError("Provided depth must be greater than 0 (got %d)" % depth)
    param_counts = defaultdict(int, {"total_params": 0, "trainable_params": 0})
    for n, p in nn_model.named_parameters():
        names = ".".join(n.split(".")[:depth])
        param_counts[names] += p.numel()
        if p.requires_grad:
            param_counts["trainable_params"] += p.numel()
        param_counts["total_params"] += p.numel()

    return param_counts


def seed_everything(seed: int) -> None:
    """Seed all random number generators."""
    random.seed(seed)  # for Python random module.
    np.random.seed(seed)  # for NumPy.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_worker_threads(num_threads: int = 2) -> None:
    """Limit the number of threads and malloc arenas for DataLoader workers.

    Two problems on high-core-count machines with many DataLoader workers:
    1. Each library (OMP, MKL, OpenBLAS, PyTorch) defaults to cpu_count threads.
       16 workers * 128 threads/library = 2000+ threads, each with its own stack.
    2. glibc malloc creates 8*cpu_count arenas per process. Freed memory in an arena
       is NEVER returned to the OS. 16 workers * 1024 arenas = TB-level fragmentation.

    Call this once at the training entry point. DataLoader workers inherit both the
    environment variables and the torch thread state via fork.

    Args:
        num_threads: Maximum number of threads per library. Default 2 is a good balance
            for DataLoader workers that mostly do I/O and lightweight compute.
    """
    import os

    str_n = str(num_threads)
    # Force override — setdefault is useless if the var was already set by
    # system profile, module init, or a previously-imported library.
    env_vars = {
        # OpenMP (used by numpy, scipy, ffmpeg, etc.)
        "OMP_NUM_THREADS": str_n,
        # Intel MKL (used by numpy/scipy when built with MKL)
        "MKL_NUM_THREADS": str_n,
        # OpenBLAS (used by numpy/scipy when built with OpenBLAS)
        "OPENBLAS_NUM_THREADS": str_n,
        # BLIS (alternative BLAS)
        "BLIS_NUM_THREADS": str_n,
        # TBB (Threading Building Blocks, used by some sklearn/numba)
        "TBB_NUM_THREADS": str_n,
        # PyTorch intra-op threads
        "TORCH_NUM_THREADS": str_n,
        # glibc malloc arena limit — the real memory fragmentation killer.
        # Default: 8 * cpu_count arenas per process (128 cores => 1024 arenas).
        # Each arena holds freed memory that is NEVER returned to the OS.
        # 16 workers * 1024 arenas * fragmented MB = TB-level waste.
        "MALLOC_ARENA_MAX": str_n,
    }
    for key, val in env_vars.items():
        os.environ[key] = val
    # Also set torch threads directly in case env var was read before we set it
    torch.set_num_threads(num_threads)
