"""Business entity resolution pipeline."""
import os

# Set before numpy is imported anywhere (this package init runs first).
# OpenBLAS otherwise reserves a buffer per CPU thread in every process and
# worker (~1 GB each here); nothing in the pipeline uses dense BLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Matches data.N_JOBS. Below the logical core count, it also stops loky from
# probing physical cores via `wmic` (absent on Windows 11), which prints a
# long harmless traceback.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
