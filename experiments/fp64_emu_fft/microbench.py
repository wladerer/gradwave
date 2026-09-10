"""Anchor the Amdahl ceiling for fp64-emulated-FFT on the RTX 3050.

Measures, on the actual hardware:
  1. native fp64 complex 3D FFT throughput on GPU (batched over bands)
  2. CPU fp64 complex 3D FFT throughput (8 threads) -- the real SCF baseline
  3. peak int8 tensor-core GEMM throughput via torch._int_mm (TOPS ceiling)
  4. PCIe H2D + D2H bandwidth for realistic band-batch tensors

All GPU timings use cuda events with warmup + sync. Deterministic shapes.
"""
import time
import torch
import numpy as np

torch.manual_seed(0)
dev = torch.device("cuda")
print("torch", torch.__version__, "dev", torch.cuda.get_device_name(0))
print("cc", torch.cuda.get_device_capability(0))

def cuda_time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms

def cpu_time(fn, iters=10, warmup=2):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms

# ------------------------------------------------------------------ FFT
print("\n== 3D complex128 FFT: GPU native fp64 vs CPU 8-thread ==")
torch.set_num_threads(8)
for N, B in [(24, 32), (48, 32), (48, 130), (64, 32), (96, 8)]:
    xg = torch.randn(B, N, N, N, dtype=torch.complex128, device=dev)
    xc = xg.cpu()
    tg = cuda_time(lambda: torch.fft.fftn(xg, dim=(-3, -2, -1)))
    tc = cpu_time(lambda: torch.fft.fftn(xc, dim=(-3, -2, -1)))
    flops = B * 5.0 * (N**3) * np.log2(N**3)  # ~5 N log2 N per complex FFT
    print(f"  N={N:3d} B={B:4d}  GPU={tg*1e3:8.1f}us  CPU8={tc*1e3:8.1f}us  "
          f"GPU/CPU={tg/tc:5.2f}x  GPUgflops={flops/(tg*1e-3)/1e9:6.1f}  CPUgflops={flops/(tc*1e-3)/1e9:6.1f}")

# ------------------------------------------------------------------ int8 GEMM peak
print("\n== int8 tensor-core GEMM peak (torch._int_mm) ==")
for M in [1024, 2048, 4096, 8192]:
    a = torch.randint(-8, 8, (M, M), dtype=torch.int8, device=dev)
    b = torch.randint(-8, 8, (M, M), dtype=torch.int8, device=dev)
    t = cuda_time(lambda: torch._int_mm(a, b))
    ops = 2.0 * M**3
    print(f"  M={M:5d}  {t*1e3:8.1f}us  {ops/(t*1e-3)/1e12:7.1f} TOPS")

# native fp64 GEMM for comparison
print("\n== native fp64 GEMM (for the crippled-fp64 ratio) ==")
for M in [1024, 2048, 4096]:
    a = torch.randn(M, M, dtype=torch.float64, device=dev)
    b = torch.randn(M, M, dtype=torch.float64, device=dev)
    t = cuda_time(lambda: a @ b)
    ops = 2.0 * M**3
    print(f"  M={M:5d}  {t*1e3:8.1f}us  {ops/(t*1e-3)/1e12:7.3f} TFLOPS fp64")

# ------------------------------------------------------------------ PCIe
print("\n== PCIe H2D / D2H bandwidth (pinned) ==")
for mb in [16, 128, 512]:
    n = mb * 1024 * 1024 // 16  # complex128 elems
    hc = torch.randn(n, dtype=torch.complex128).pin_memory()
    dc = torch.empty(n, dtype=torch.complex128, device=dev)
    th = cuda_time(lambda: dc.copy_(hc, non_blocking=True), iters=20)
    dd = torch.randn(n, dtype=torch.complex128, device=dev)
    ho = torch.empty(n, dtype=torch.complex128).pin_memory()
    td = cuda_time(lambda: ho.copy_(dd, non_blocking=True), iters=20)
    print(f"  {mb:4d}MB  H2D={mb/1024/(th*1e-3):6.1f} GB/s  D2H={mb/1024/(td*1e-3):6.1f} GB/s")
print("DONE")
