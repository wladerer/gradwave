"""One-liner third angle: is cuSOLVER zheevd on the 3050 fp64-ALU-bound or
memory-bound at Davidson subspace dims (<= 4*nb)?"""
import time
import torch

dev = torch.device("cuda")
torch.set_num_threads(8)

def cuda_time(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us

def cpu_time(fn, iters=10, warmup=2):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e6

print("dim   GPU_zheevd_us   CPU8_us   GPU_eff_GFLOPs (9n^3 complex-op est)")
for n in [64, 96, 128, 256, 520, 1024]:
    a = torch.randn(n, n, dtype=torch.complex128, device=dev)
    h = (a + a.conj().T) / 2
    hc = h.cpu()
    tg = cuda_time(lambda: torch.linalg.eigh(h))
    tc = cpu_time(lambda: torch.linalg.eigh(hc))
    flops = 9.0 * n**3  # rough zheevd flop count (complex tridiag + back-transform)
    print(f"{n:5d} {tg:12.0f} {tc:10.0f} {flops/(tg*1e-6)/1e9:12.2f}")
print("fp64 GEMM peak (measured elsewhere) = 130 GFLOPs; mem-bound est for n^2*16B/(192GB/s) is negligible -> compare eff rate to 130 to judge ALU-bound")
print("DONE")
