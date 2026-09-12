"""SCF option objects (refactor stage 1; grouped options stage 2).

`scf_uspp` accumulated ~20 keyword parameters, each individually
justified and collectively unreadable — and `scf.loop.scf` grew to ~37.
The options are grouped by what they control, frozen (an SCF run's
configuration is immutable), and constructed from plain kwargs for
backward compatibility, so
`scf_uspp(system, xc, etol=1e-9, mixing_scheme="johnson")` keeps working
while `scf_uspp(system, xc, opts=SCFOptions(...))` becomes the readable
form. Defaults here are THE defaults for the opts path; both drivers
also accept `opts` OR the flat kwargs, never a conflicting mix (each
driver's guard rejects a non-default flat kwarg alongside `opts`).

Stage 2 adds the grouped sub-objects shared by both drivers:

* ``MemoryOptions``   — the ``scf.memory`` footprint/wall-time knobs
  (k-streaming, k-parallel, Davidson subspace, dense-box budget). These
  now flow to the solver as ARGUMENTS (see solvers/davidson.py's
  per-solve resolvers) instead of the retired api → environment bridge;
  the ``GRADWAVE_*`` env vars remain the documented user-facing
  overrides, read once per solve where each knob is resolved.
* ``BoundaryOptions`` — ESM open-boundary electrostatics + constant-µ.
* ``SpinOptions``     — collinear spin count, moment seed, fixed moment.
* ``HubbardOptions``  — the DFT+U convergence aids (the manifold list
  itself stays a runtime argument, like `start_from`/`dist_ctx`).

Each group defaults to ``None`` on ``SCFOptions`` (= "not specified via
opts": the flat kwargs govern, byte-for-byte the historical behaviour);
a provided group takes over its kwargs, and the driver rejects a
conflicting non-default flat kwarg. Fields a formalism does not support
are rejected loudly by that driver (e.g. ``target_mu`` / ``k_chunk`` on
the USPP path), never silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class MixerOptions:
    scheme: str | None = None  # None → per-nspin default (johnson nspin=1,
    # pulay nspin=2, resolved in scf_uspp); or pulay | broyden | johnson
    alpha: float = 0.7
    history: int | None = None  # None → per-scheme default (johnson 12, else 8)
    kerker: bool | None = None  # None → on for smeared systems
    precond: str = "kerker"  # kerker | local_tf (position-dependent TF screening)
    metric: str = "plain"  # plain | coulomb (johnson only)
    w0: float = 0.01  # johnson regularization
    trust_factor: float = 20.0
    adapt_step: bool = False  # opt-in collapse protection (see docs/manual/wisdom.md)
    spin_precond: bool = False  # Stoner m-channel preconditioner
    # None → per-scheme default: 1.0 for johnson, 0.4 otherwise. The 0.4
    # damping of the on-site becsum↔ddd mode is a Pulay-era stabilizer;
    # Johnson's normalized multisecant handles that mode natively and the
    # damping just brakes the composite (FM Ni: 27 it at 0.4 → 16 at 1.0)
    bec_step_scale: float | None = None


@dataclass(frozen=True)
class MemoryOptions:
    """The ``scf.memory`` block (inputs.models.MemoryParams), as solver-bound
    options. All opt-in; every ``None`` leaves the run byte-for-byte the
    historical behaviour (or defers to the matching ``GRADWAVE_*`` env var,
    which remains the documented user-facing override and is layered over
    these values at solve time). Norm-conserving path unless noted; the USPP
    driver consumes only ``dense_budget_gb`` and rejects the rest."""

    k_chunk: int | None = None  # k-streaming: k-points per Davidson call
    k_parallel: int | None = None  # per-k thread-pool eigensolve (CPU only)
    max_dim_factor: int | None = None  # FORCE the Davidson subspace multiple (≥ 2)
    subspace_budget_gb: float | None = None  # auto-gate factor toward 2 under budget
    subspace_storage: str | None = None  # None/"complex128" | "complex64" (V/HV bytes ×0.5)
    dense_budget_gb: float | None = None  # CPU dense FFT-box band-chunk cap [GB]


@dataclass(frozen=True)
class BoundaryOptions:
    """Electrostatic boundary (slab geometry, c ⊥ a,b) + constant-potential."""

    kind: str = "periodic"  # periodic | open_z | open_z_metal (ESM)
    esm_bias: float = 0.0  # applied capacitor bias [V] for open_z_metal
    target_mu: float | None = None  # grand-canonical SCF (NC only): fix µ [eV], float N


@dataclass(frozen=True)
class SpinOptions:
    """Collinear spin configuration."""

    nspin: int = 1
    start_mag: list[float] | None = None  # per-species OR per-atom moment fractions
    tot_magnetization: float | None = None  # fixed spin moment M=N↑−N↓ (NC only)


@dataclass(frozen=True)
class HubbardOptions:
    """DFT+U convergence aids (the HubbardManifold list itself is a runtime
    argument on the driver, not configuration)."""

    occ_mix: float = 1.0  # occupation-matrix damping β in (0,1]; 1.0 = raw lag
    u_ramp_iters: int = 0  # linear U ramp length; 0 = off
    alpha: list[float] | None = None  # per-site rigid potential α [eV] (NC only)


@dataclass(frozen=True)
class SCFOptions:
    smearing: str = "none"
    width: float = 0.1
    max_iter: int = 60
    etol: float = 1e-8
    rhotol: float = 1e-7
    diago_tol: float = 1e-9
    criterion: str = "drho"  # drho | energy
    # opt-in energy-metric convergence gate: converge on the residual's exact
    # second-order energy error 1/2<r|K_Hxc|r> < entol instead of rhotol (etol
    # and the stale-solve guard unchanged), overriding `criterion`. The honest
    # criterion for metallic magnets. False (default) leaves the density gate
    # bit-for-bit unchanged.
    energy_metric: bool = False
    entol: float = 1e-6  # eV, the energy-error threshold for energy_metric
    rho_safety: float = 1e-2
    batched: bool = True
    # fp32 draft for the batched Davidson while the diago tolerance is
    # loose (> 1e-5); subspace algebra and every SCF quantity stay fp64,
    # so converged results are unchanged. Opt-in; the payoff is on GPUs
    # where consumer fp64 throughput is 1/64 of fp32.
    mixed_precision: bool = False
    verbose: bool = True
    # block eigensolver (norm-conserving standard problem only): auto |
    # davidson | chebyshev | davidson-native. The USPP/PAW generalized
    # S-metric driver rejects an explicit "chebyshev".
    eigensolver: str = "auto"
    mixer: MixerOptions = field(default_factory=MixerOptions)
    # Grouped sub-options. None = "not specified via opts": the driver's flat
    # kwargs govern that concern, byte-for-byte the historical behaviour. A
    # provided group takes over, and a conflicting non-default flat kwarg
    # raises (same opts-vs-flat guard as the scalar fields).
    memory: MemoryOptions | None = None
    boundary: BoundaryOptions | None = None
    spin: SpinOptions | None = None
    hubbard: HubbardOptions | None = None

    @classmethod
    def from_kwargs(cls, **kw) -> SCFOptions:
        """Build from flat legacy kwargs (mixing_alpha=..., etc.).
        Unknown keys raise, misspelled tolerances must not pass silently."""
        rename = {
            "mixing_alpha": "alpha", "mixing_history": "history",
            "mixing_scheme": "scheme", "mixing_kerker": "kerker",
            "mixing_metric": "metric",
        }
        mix_names = {f.name for f in fields(MixerOptions)}
        scf_names = {f.name for f in fields(SCFOptions)} - {"mixer"}
        mix_kw, scf_kw = {}, {}
        for key, val in kw.items():
            name = rename.get(key, key)
            if name in mix_names:
                mix_kw[name] = val
            elif name in scf_names:
                scf_kw[name] = val
            else:
                raise TypeError(f"unknown SCF option {key!r}")
        return cls(mixer=MixerOptions(**mix_kw), **scf_kw)
