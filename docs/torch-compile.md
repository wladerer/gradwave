# torch.compile in gradwave: where it pays and where it does not

`torch.compile` is not a general accelerator for this code, and the
two earlier attempts that concluded that were right about the slice they measured.
But that verdict was reached on the complex, FFT-bound part of the stack, and it was
generalized too far. There is one real-valued slice nobody isolated, the
exchange-correlation functional, and on that slice Inductor gives a large,
bit-accurate speedup. This note records both halves so the next person does not have
to re-derive them.

## The part where it does not help, and why that is settled

The SCF hot path is the Hamiltonian apply inside the batched Davidson solver. On the
profiled O2 case that is 41% of the SCF, and it is dominated by two complex128 FFTs
per apply. Two facts kill `torch.compile` there.

- Inductor does not codegen complex operations. The entire core carries
  `complex128` wavefunctions and `complex128` scatter and einsum work, so Dynamo
  either graph-breaks or falls back, and compiling a complex kernel measured slower
  than eager.
- The FFTs are opaque library calls (cuFFT on GPU, pocketfft on CPU). A compiler
  cannot fuse across them or improve them, and they are the majority of the apply.

A real-decomposed spin-mix kernel compiled to 1.69x on its own
micro-benchmark, but as a share of the FFT-bound apply it moved GaAs SOC from 250 to
248 s, which is noise. The related experiment of capturing the apply in a CUDA graph
replayed at 1.0 to 1.1x eager, which says the kernels are already back-to-back and
there is no launch gap to remove. The binding constraint on the small-system GPU
path is fp64 throughput, not dispatch. All of this is in `docs/manual/performance.md`
under "What does not help", and the entry there is correct for the Hamiltonian apply.

The prerequisite that would open the apply to a compiler is a full complex-to-real
decomposition of the core, real and imaginary parts carried separately through
`core/hamiltonian.py` and the projector einsums. That is more invasive than the
Gamma-point real-wavefunction specialization already deferred for a 1.3 to 1.5x
ceiling, and it still would not touch the FFTs. It is not worth it.

## The part that was never measured: exchange-correlation

Both earlier attempts compiled complex kernels. The XC functional is real-valued. It
takes a real-space density (and `|grad rho|^2` for GGAs) and runs a chain of roughly
thirty elementwise operations, several of them transcendental: a cube root for the
Fermi wavevector, `exp`, `log1p`, `sqrt`, and integer powers. That chain is exactly
what Inductor fuses well, and `v_xc = dE_xc/d rho` comes from one autograd backward
through the same chain, which the compiler fuses too.

Measured on the laptop (8 CPU threads, `PBE.energy_density` on a 64^3 grid,
float64):

| path | eager | compiled | speedup |
|---|---|---|---|
| forward energy density | 65 ms | 8 ms | 8.2x |
| forward + v_xc backward | ~600 ms | ~20 ms | ~30x |

The compiled `v_xc` matches eager to `5e-15`, machine precision. The large
forward-plus-backward number reflects how heavy the eager backward of a
transcendental chain is. The compiler fuses the backward graph as well, which is
where most of the win comes from.

Useful and low risk, but the microbenchmark overstates the end-to-end effect. On the
O2 profile XC is
about 26% of the SCF (5.8 of 22 s), and a large fraction of that 5.8 s is the
FFT-based gradient and divergence assembly in `core/density.py`, which a compiler
does not touch. The pointwise `energy_density` math itself is the compilable part.
So the realistic end-to-end gain on an ordinary ground-state SCF is in the low tens
of percent of the XC share, call it a few percent of the whole, not 8x and not 30x.

## Where the XC win actually concentrates

The pointwise share is small in one plain SCF, but three workloads call the XC
transcendental chain far more than once per iteration, and they are CPU-bound rather
than FFT-bound. These are the real targets.

- The PAW one-center quadrature (`scf/paw_onsite.py`) calls `energy_density` inside a
  loop over angular grid points, many small real-valued calls per SCF iteration on
  the radial-times-angular grid. Fusing removes per-call dispatch overhead on top of
  the arithmetic, which is where small-tensor eager mode is weakest.
- The response and HVP machinery. `f_xc` is a double backward through the XC energy,
  and it is called repeatedly by the density-loss adjoint (`postscf/uspp_implicit.py`),
  the Newton finisher (`scf/newton.py`), the dielectric Sternheimer
  (`postscf/dielectric.py`), the Stoner preconditioner (`scf/spin_precond.py`), and
  the learned-U response (`postscf/hubbard_u.py`). These are transcendental-heavy and
  do not FFT.
- Learned-XC training (`examples/train_xc_paw.py`, `core/xc/learnable.py`). This is
  the strongest case. Training runs the XC forward, `v_xc`, and the `f_xc` HVP many
  times per epoch at roughly 3 to 6 minutes per epoch, all on the real grid, and the
  functional parameters are on the graph the whole time. A compiled XC layer attacks
  exactly the part of training that is not FFT.

## The single clean insertion point

`XCFunctional.energy_density` (`core/xc/base.py`) is the one choke point. Every
functional subclasses it, and `energy()` and every caller above route through it. A
compiled path can wrap that method behind an opt-in flag, cache one compiled callable
per functional instance, and fall back to eager on any compile error. Nothing else
in the stack has to change. The learnable functionals need the parameters treated as
graph inputs rather than baked constants, which they already are, so guard recompiles
do not fire every training step.

## What has to change for this to be usable

- NixOS toolchain. Inductor's CPU C++ codegen shells out to a host compiler and
  needs `openssl` on `PATH` for its cache hashing, which the managed venv does not
  provide by default. The first compile fails with a bare
  `FileNotFoundError: openssl`. The GPU path additionally needs
  `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`, since there is no `/sbin/ldconfig`.
  Both are environment issues, not code issues, but a compiled path must degrade to
  eager gracefully when the toolchain is absent so a stock checkout never breaks.
- Dynamic shapes. The grid size changes across systems and across the PAW angular
  quadrature, and each new shape triggers a recompile. Compiling two grid sizes in
  one process was enough to blow a two-minute budget on compile time alone. Use
  `dynamic=True`, or accept that the compile cost amortizes only over a long SCF or a
  training run, and never inside a short one-shot.
- Double backward. The `f_xc` HVP compiles and stays bit-accurate in the forward and
  first backward, but double backward through compiled code is the fragile path and
  needs its own regression gate against the eager HVP before any response or training
  code depends on it.
- Scope. This does nothing for the complex, FFT-bound majority of a normal
  SCF. It is a targeted win for XC-heavy, CPU-bound workloads, learned-XC training
  first, the PAW one-center loop and the response HVPs second. Describe it as that,
  not as a general speedup, so it does not get benchmarked on a plain Si SCF and
  dismissed the way the complex attempts were.

## Recommendation

Add an opt-in `compile_xc` flag that wraps `XCFunctional.energy_density` in a cached
compiled callable with an eager fallback, gate it with a bit-accuracy test on
`v_xc` and the `f_xc` HVP, and point it at the learned-XC training loop first. Leave
the Hamiltonian apply alone; that verdict is settled. And correct the blanket
"tried and removed" line in `docs/manual/performance.md` to say the complex apply
was tried and removed while the real-valued XC layer is a live win, so the next
reader does not inherit the overgeneralization this note exists to fix.

## Implemented, and one correction

Landed as the `compile_xc` flag on the `GradWave` calculator, backed by
`CompilableXC` and `_DoubleSafeXC` in `core/xc/base.py`, with the bit-accuracy gate
in `tests/unit/test_xc_compile.py`. Measured here at 64³ float64, 8 threads, the
compiled forward is 19x eager and forward-plus-`v_xc` is 16x, `v_xc` bit-accurate to
3e-16. Numbers and scope are in `docs/manual/performance.md` under "Compiled XC
layer".

The gate forced one correction. This PyTorch's aot_autograd cannot double-backward
through compiled code, it raises `does not currently support double backward`, not a
slow-but-correct result. So the `f_xc` HVP does not compile at all, and the hope
above that response HVPs and HVP-based learned-XC training would benefit was wrong.
Only the forward and the first-order `v_xc` compile. `_DoubleSafeXC` detects the
second-derivative path with `torch.is_grad_enabled()` inside its backward and routes
it to eager, so response and training code stays correct with the flag on, it just
does not accelerate on the HVP leg. The forward and `v_xc` legs of training still
win, which keeps learned-XC the strongest target, just for a narrower reason.
