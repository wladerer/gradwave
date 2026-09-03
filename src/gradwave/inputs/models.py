"""The input schema: typed dataclass settings objects (see load_input)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms

if TYPE_CHECKING:
    pass

class InputError(ValueError):
    """A malformed input file. Carries the input path (prepended by
    ``load_input``) so the message points at the file the user edited."""


@dataclass(frozen=True)
class MixingParams:
    scheme: str = "auto"  # auto | pulay | broyden | johnson. auto (the default)
    # defers to each formalism's evidence-backed resolver (johnson for USPP/PAW
    # and for collinear-spin nspin=2 norm-conserving, pulay otherwise); an
    # explicit scheme always wins. See scf.loop._resolve_mixing_scheme and
    # scf.uspp_loop._resolve_uspp_mixing_scheme.
    alpha: float = 0.7
    history: int | None = None  # None → per-scheme default (johnson 12, else 8)
    kerker: str | bool = "auto"  # auto: on iff smearing enabled
    precond: str = "kerker"  # kerker | local_tf. local_tf is the position-


    # dependent Thomas-Fermi preconditioner; it replaces the constant Kerker
    # filter on the charge channel, so `kerker` no longer applies there.
    chi0_precond: bool = False  # opt-in subspace-χ₀ Woodbury dielectric
    # preconditioner (norm-conserving path only). Built once at the first
    # converged SCF of a multi-geometry task (relaxation) and reused on every
    # later step, so its one-time cost amortizes; auto-abstains to `precond`
    # above on homogeneous/well-conditioned cells (the ρ(M) screening-eigenvalue
    # gate — see scf.subspace_chi0.Chi0PrecondCache). Default off. Targets
    # inhomogeneous metals/surfaces/slabs (the M1 crux regime).


@dataclass(frozen=True)
class MagneticParams:
    """Magnetic-channel SCF controls for the noncollinear/spinor path
    (task: scf with noncollinear: true). The spinor driver resolves its
    (ρ, m⃗) mixing independently of scf.mixing, so these are the only knobs that
    reach it. Ignored by the collinear formalisms, whose magnetization mixing is
    set by scf.mixing. See docs/manual/noncollinear-soc.md."""

    mixer: str = "pulay"  # pulay | johnson | broyden — the (ρ, m⃗) mixer class
    spin_precond: bool = False  # Stoner preconditioner on the longitudinal m⃗ channel
    mixing_alpha: float | None = None  # separate step for the m⃗ blocks; None keeps
    # the scheme-dependent driver default (max(scf.mixing.alpha, 0.6) for pulay/
    # broyden, 0.3 for johnson — see scf.noncollinear._resolve_mag_mixing_alpha)
    diago_schedule: str = "linear"  # linear | quadratic adaptive diago-tol schedule

    def __post_init__(self):
        if self.mixer not in ("pulay", "johnson", "broyden"):
            raise InputError(
                "scf.magnetic.mixer must be 'pulay', 'johnson', or 'broyden', "
                f"got {self.mixer!r}")
        if self.diago_schedule not in ("linear", "quadratic"):
            raise InputError(
                "scf.magnetic.diago_schedule must be 'linear' or 'quadratic', "
                f"got {self.diago_schedule!r}")


@dataclass(frozen=True)
class SCFParams:
    max_iter: int = 100
    etol: float = 1.0e-7
    rhotol: float = 1.0e-7
    mixing: MixingParams = field(default_factory=MixingParams)
    diago_tol: float = 1.0e-9
    # write the per-iteration SCF flight-recorder trace to scf_trace.json (the
    # cheap diagnostics are always summarized into the output regardless). Off
    # by default to keep the output directory lean.
    trace: bool = False
    # convergence criterion: "density" (default) gates on the density residual
    # (< rhotol) plus the energy tail (< etol); "energy" gates on the residual's
    # exactly-computed second-order energy error (1/2<r|K_Hxc|r> < entol) plus
    # the same energy tail, and is the honest criterion for metallic magnets
    # whose magnetization-channel density residual floors above any reachable
    # rhotol (docs/manual/convergence.md).
    convergence: str = "density"
    entol: float = 1.0e-6  # eV, energy-error threshold used when convergence="energy"
    # block eigensolver: "auto" (size-gated: CheFSI for large-N slabs, else
    # Davidson), "davidson" (the workhorse), or "chebyshev" (Chebyshev-filtered
    # subspace iteration, forced). CheFSI is norm-conserving only; the USPP/PAW
    # generalized S-metric problem rejects an explicit chebyshev choice, and
    # "auto" resolves to davidson there (see api / calculator).
    eigensolver: str = "auto"  # auto | davidson | chebyshev
    # electrostatic boundary (slab geometry, c ⊥ a,b): "periodic" (default) |
    # "open_z" (open-boundary / ESM vacuum both sides — no dipole correction,
    # box-independent surfaces) | "open_z_metal" (metal Dirichlet planes at both
    # z-box edges — a capacitor; pair with esm_bias for field-effect). See
    # core/energies/esm.py.
    boundary: str = "periodic"
    esm_bias: float = 0.0  # applied capacitor bias [V] for boundary="open_z_metal"
    target_mu: float | None = None  # constant-potential (grand-canonical) SCF: hold
    # the Fermi level µ [eV] fixed and let the electron count float. Requires
    # boundary="open_z_metal" and a smearing. None = ordinary fixed-N SCF.
    # magnetic-channel SCF controls for the noncollinear/spinor path only
    magnetic: MagneticParams = field(default_factory=MagneticParams)

    def __post_init__(self):
        if self.convergence not in ("density", "energy"):
            raise InputError(
                "scf.convergence must be 'density' or 'energy', got "
                f"{self.convergence!r}")
        if self.eigensolver not in ("auto", "davidson", "chebyshev"):
            raise InputError(
                "scf.eigensolver must be 'auto', 'davidson' or 'chebyshev', got "
                f"{self.eigensolver!r}")
        if self.boundary not in ("periodic", "open_z", "open_z_metal"):
            raise InputError(
                "scf.boundary must be 'periodic', 'open_z' or 'open_z_metal', got "
                f"{self.boundary!r}")


@dataclass(frozen=True)
class SlabParams:
    """Density-tail vacuum auto-sizer for open-boundary (ESM) slabs.

    ESM makes the open-axis electrostatics box-independent, so the vacuum only
    has to hold the physical density tail (~4–6 Å/face), not decouple periodic
    images (the habitual 10–20 Å). Trimming the open (c) axis to the SAD-density
    tail plus a margin shrinks both the FFT box ``Nz`` and the plane-wave count
    ``npw`` (both ∝ open-axis length), cutting the FFT H-apply and the O(nb²·npw)
    Rayleigh–Ritz together. The box is a forward hyperparameter (set once, frozen
    for the solve — like ``ecut``), so there is no autograd interaction.
    Opt-in; a no-op unless ``scf.boundary`` is an open mode and both gates pass.
    See ``api._slab.resolve_slab_box``.
    """

    vacuum_autosize: bool = False  # trim the open axis to the SAD density tail
    # margin discipline keyed to the requested observable: "energy"/"forces" are
    # tail-limited (aggressive ρ<1e-4, thin margin); "workfunction"/dipole are
    # plateau-limited (vacuum_level needs a flat plateau — larger margin). The
    # conservative "workfunction" is the default.
    vacuum_target: str = "workfunction"  # energy | workfunction
    vacuum_tol: float | None = None  # override the tail tolerance [e/Å³]
    vacuum_margin: float | None = None  # override the vacuum margin [Å]
    min_vacuum: float = 3.0  # safety floor of vacuum per face [Å] (warns if it binds)
    # gates: below either, the box is left exactly as the user set it. A slab
    # satisfies both; a small bulk-ish cell violates them (a clean flip).
    npw_gate: int = 8000  # only trim when arithmetic-bound (npw ≳ this)
    vacuum_fraction_gate: float = 0.3  # only trim when vacuum-dominated

    def __post_init__(self):
        if self.vacuum_target not in ("energy", "workfunction"):
            raise InputError(
                "slab.vacuum_target must be 'energy' or 'workfunction', got "
                f"{self.vacuum_target!r}")
        if self.vacuum_tol is not None and self.vacuum_tol <= 0.0:
            raise InputError(
                f"slab.vacuum_tol must be > 0, got {self.vacuum_tol}")
        if self.min_vacuum < 0.0:
            raise InputError(
                f"slab.min_vacuum must be ≥ 0, got {self.min_vacuum}")


@dataclass(frozen=True)
class SmearingParams:
    type: str = "none"  # none | fermi-dirac | gaussian | mp1 | cold
    width: float = 0.1  # eV


@dataclass(frozen=True)
class KPointsParams:
    mesh: tuple[int, int, int] = (1, 1, 1)
    shift: tuple[int, int, int] = (0, 0, 0)
    # target reciprocal-space k-spacing [Å⁻¹]; when set (not None), the mesh is
    # derived at build time from the cell via kpoints.slab_kmesh — anisotropic and
    # slab-aware (the detected vacuum axis is pinned to a single Γ point). Overrides
    # `mesh`. Pair with a Γ-centered shift (0,0,0). None → use `mesh` verbatim.
    kspacing: float | None = None

    def __post_init__(self):
        if self.kspacing is not None and self.kspacing <= 0.0:
            raise InputError(
                f"kpoints.kspacing must be positive, got {self.kspacing!r}")


@dataclass(frozen=True)
class RelaxParams:
    # bfgs is the default: on displaced diamond it needs 3 steps where
    # fire needs 25 (measured 2026-07-15); fire remains available for
    # far-from-minimum or noisy-force cases
    optimizer: str = "bfgs"  # bfgs | fire
    fmax: float = 0.01  # eV/Å (also gates stress under cell relaxation)
    max_steps: int = 200
    cell: bool = False  # variable-cell: relax the lattice with the atoms (stress)
    pressure: float = 0.0  # external hydrostatic pressure [GPa]; cell relaxation only
    # relaxation engine: "nested" runs a full SCF inside every ASE geometry step
    # (the robust default, all formalisms/spin/metals); "joint" descends on
    # (strain, positions, orbitals) at once via opt/joint.py — far fewer H-applies
    # but NC insulators only, and it falls back to "nested" on non-convergence or
    # an unsupported system (USPP/PAW, metal, odd electron count).
    # "newton" is the exact-Hvp Steihaug trust-region Newton-CG engine
    # (opt/newton.py); same NC-insulator contract and nested fallback as "joint".
    method: str = "nested"  # nested | joint | newton
    # Pulay (basis-incompleteness) pressure correction for variable-cell relax:
    # the fixed-basis stress under-pressures a too-small basis, silently driving
    # soft cells toward spuriously small volumes (#217). None (auto, default)
    # enables it whenever cell relaxation is on and the estimator supports the
    # run (norm-conserving, symmetry: false, Γ-centered mesh, no DFT+U,
    # scalar-relativistic — see postscf/stress_error.py); true requires it
    # (InputError when unsupported); false disables. Never applied to a
    # fixed-cell relax.
    pulay_correction: bool | None = None
    # Charge-density (and USPP becsum) extrapolation across ionic steps, QE-style:
    # each nested step seeds the SCF from an extrapolation of the previous
    # geometries' converged states instead of re-converging cold. "reuse" (the
    # default, unchanged behaviour) shifts the atomic-superposition part with the
    # atoms and reuses the bonding remainder; "linear"/"quadratic" additionally
    # extrapolate that remainder from the last two/three steps by least-squares
    # matching the new positions; "none" disables the warm start (cold SAD every
    # step). Nested engine, nspin=1 only — a spin-polarized or joint/newton run
    # ignores it. See docs/manual/geometry-optimization.md.
    extrapolation: str = "reuse"  # none | reuse | linear | quadratic
    # complement-correction solver for the Pulay (basis-incompleteness) pressure
    # estimate, inert unless the correction is on (see pulay_correction above):
    # "diagonal" (the default, kinetic-only resolvent) or "cg" (preconditioned
    # iterative annulus solve, which recovers a larger fraction of the true Pulay
    # pressure for a modest per-step cost — see docs/manual/geometry-optimization.md).
    pulay_solver: str = "diagonal"  # diagonal | cg
    # ------------------------------------------------------------------
    # Parallel / speculative line search (opt-in, nested engine only).
    # A quasi-Newton relax computes a search direction dₖ each ionic step, then
    # takes a fixed step along it (BFGS uses α=1, capped by maxstep) — the step
    # length is not searched. This replaces that fixed step with ONE parallel
    # round: evaluate Rₖ + αᵢ·dₖ for a spread of αᵢ CONCURRENTLY (each a forward
    # SCF warm-started from Rₖ's checkpoint, returning E and F), fit a cubic
    # through the (αᵢ, Eᵢ, gᵢ = −Fᵢ·d̂) samples and step to the interpolated
    # minimum α*. It only changes the PATH: the accepted geometry is always
    # re-evaluated by the main calculator at full SCF tol (the next ionic step's
    # force call), so the relaxed minimum matches a plain serial BFGS relax to
    # the geometry/energy tolerance — same minimum, different route.
    #   "off"      (default) serial fixed step — BYTE-IDENTICAL to today.
    #   "parallel" fan candidates out every ionic step.
    #   "adaptive" fan out ONLY when the optimizer is struggling (energy rose on
    #              the last accepted step, or max|F| failed to decrease over the
    #              last `line_search_patience` steps); a normal serial step
    #              otherwise.
    # Scope: positions-only relax (cell: false) with the bfgs optimizer; a
    # cell/fire/joint/newton run silently keeps the serial step (recorded in the
    # relax block). FORWARD-ONLY — the candidate SCFs run in worker processes and
    # PyTorch autograd graphs do not cross a process boundary, so a
    # DIFFERENTIABLE relax must leave this "off".
    line_search: str = "off"  # off | parallel | adaptive
    # DECOUPLED knobs: n_samples is how many α to try (the cubic sample count);
    # n_workers is how many run AT ONCE. PEAK memory ≈ n_workers × one-SCF and is
    # INDEPENDENT of n_samples — more samples than workers run in later waves, not
    # concurrently. Cap RAM via n_workers; raise n_samples to resolve the cubic.
    line_search_n_samples: int = 3
    line_search_n_workers: int = 2
    # Explicit geometric α schedule around the proposed step (α=1). Empty →
    # auto-generate n_samples points geometrically bracketing 1.0 (e.g. n=4 →
    # {0.25, 0.5, 1.0, 2.0}). α* is clamped to line_search_max_alpha: a large α
    # crosses a bigger geometry change → a worse warm-start seed → a costlier
    # candidate SCF, so the step is bounded.
    line_search_alphas: tuple[float, ...] = ()
    line_search_max_alpha: float = 2.5
    # adaptive trigger window: struggle is judged over the last `patience`
    # accepted steps (max|F| failed to decrease across them, or E increased).
    line_search_patience: int = 1
    # PREDICTIVE trigger: search the first `line_search_warmup` ionic steps
    # unconditionally (0 = off). The BFGS Hessian is least informed early, so the
    # step is likeliest to overshoot then; searching those steps up front pre-empts
    # the first overshoot instead of reacting to it (adaptive fires only after the
    # damage). Applies to `adaptive`; `parallel` already searches every step.
    line_search_warmup: int = 0
    # DENSER bracket during warmup only: warmup steps use this many α samples
    # (0 → inherit line_search_n_samples). Early overshoot needs a finer/wider
    # search to pin the true minimum, but paying for extra samples every step is
    # wasteful — so spend them only on the warmup steps where they earn their keep.
    line_search_warmup_samples: int = 0
    # Initial BFGS Hessian: "identity" (ASE default, scaled unit matrix) or "lindh"
    # (a cheap curvature-aware model Hessian — stiff bonds, soft non-bonded — that
    # removes the early overshoot on stiff/soft-mixed systems). See opt.model_hessian.
    initial_hessian: str = "identity"
    # Outer-SCF tolerance ladder (nested engine only, opt-in): schedule each
    # ionic step's SCF stop tolerance from the optimizer's current max|F| — loose
    # when forces are large (early throwaway geometries), tightening as forces
    # shrink — then re-solve the converged geometry at the full rhotol so the
    # reported/differentiated result sits at the exact fixed point. Off by default
    # (behaviour byte-identical to a constant-rhotol relax). Never applied to
    # EOS/phonon/elastic (their per-config observables can't be loosened).
    tol_ladder: bool = False
    # rhotol_step = clamp(c * max|F|**p, lo=rhotol_final, hi=rhotol_start), with
    # etol scaled proportionally. p=2 (quadratic) tightens faster than p=1
    # (linear); c sets the overall scale. rhotol_final None → the scf.rhotol
    # (so the exactness re-solve lands on the same fixed point as a baseline run).
    tol_ladder_c: float = 1.0e-3
    tol_ladder_p: float = 2.0
    tol_ladder_rhotol_start: float = 1.0e-4  # first/loosest step tol (hi clamp)
    tol_ladder_rhotol_final: float | None = None  # None → scf.rhotol (lo clamp)
    # first ionic step (no previous fmax yet): "loose" applies rhotol_start to
    # step 1 too; "tight" solves step 1 at rhotol_final and ladders from step 2.
    tol_ladder_first_step: str = "loose"  # loose | tight

    def __post_init__(self):
        # YAML parses the bare word `off` (also `no`/`false`) as the boolean False
        # (the "Norway problem"), so a natural `line_search: off` arrives here as
        # False. Coerce it back to the string "off" so the unquoted spelling — the
        # one shown in the template — just works.
        if self.line_search is False:
            object.__setattr__(self, "line_search", "off")
        if self.method not in ("nested", "joint", "newton"):
            raise InputError(
                "relax.method must be 'nested', 'joint', or 'newton', got "
                f"{self.method!r}")
        if self.extrapolation not in ("none", "reuse", "linear", "quadratic"):
            raise InputError(
                "relax.extrapolation must be 'none', 'reuse', 'linear', or "
                f"'quadratic', got {self.extrapolation!r}")
        if self.pulay_solver not in ("diagonal", "cg"):
            raise InputError(
                "relax.pulay_solver must be 'diagonal' or 'cg', got "
                f"{self.pulay_solver!r}")
        if self.line_search not in ("off", "parallel", "adaptive"):
            raise InputError(
                "relax.line_search must be 'off', 'parallel', or 'adaptive', "
                f"got {self.line_search!r}")
        if self.line_search_n_samples < 2:
            raise InputError(
                "relax.line_search_n_samples must be >= 2 (a cubic fit needs at "
                f"least two E/g samples), got {self.line_search_n_samples}")
        if self.line_search_n_workers < 1:
            raise InputError(
                "relax.line_search_n_workers must be >= 1, got "
                f"{self.line_search_n_workers}")
        if self.line_search_max_alpha <= 0.0:
            raise InputError(
                "relax.line_search_max_alpha must be > 0, got "
                f"{self.line_search_max_alpha}")
        if self.line_search_patience < 1:
            raise InputError(
                "relax.line_search_patience must be >= 1, got "
                f"{self.line_search_patience}")
        if self.line_search_warmup < 0:
            raise InputError(
                "relax.line_search_warmup must be >= 0, got "
                f"{self.line_search_warmup}")
        if self.line_search_warmup_samples < 0:
            raise InputError(
                "relax.line_search_warmup_samples must be >= 0, got "
                f"{self.line_search_warmup_samples}")
        if self.initial_hessian not in ("identity", "lindh"):
            raise InputError(
                "relax.initial_hessian must be 'identity' or 'lindh', got "
                f"{self.initial_hessian!r}")
        object.__setattr__(
            self, "line_search_alphas",
            tuple(float(a) for a in self.line_search_alphas))
        if any(a <= 0.0 for a in self.line_search_alphas):
            raise InputError(
                "relax.line_search_alphas must all be > 0, got "
                f"{self.line_search_alphas}")
        if self.tol_ladder_first_step not in ("loose", "tight"):
            raise InputError(
                "relax.tol_ladder_first_step must be 'loose' or 'tight', got "
                f"{self.tol_ladder_first_step!r}")
        if self.tol_ladder:
            if self.tol_ladder_c <= 0.0 or self.tol_ladder_p <= 0.0:
                raise InputError(
                    "relax.tol_ladder_c and tol_ladder_p must be positive")
            if self.tol_ladder_rhotol_start <= 0.0:
                raise InputError(
                    "relax.tol_ladder_rhotol_start must be positive")
            rf = self.tol_ladder_rhotol_final
            if rf is not None and (rf <= 0.0 or rf > self.tol_ladder_rhotol_start):
                raise InputError(
                    "relax.tol_ladder_rhotol_final must be in "
                    "(0, tol_ladder_rhotol_start]")


@dataclass(frozen=True)
class BandsParams:
    path: str = ""  # ASE bandpath string, e.g. "LGXUG"; empty = ASE default
    npoints: int = 120
    nbands: int | None = None
    irreps: bool = False  # label bands at special points with Mulliken symbols


@dataclass(frozen=True)
class OpticsParams:
    """Independent-particle / RPA optical dielectric function ε(ω) over the SCF
    k-mesh (insulators, norm-conserving, nspin=1)."""
    omega_max: float = 20.0   # eV — top of the ω grid
    n_omega: int = 600        # points on the ω grid
    eta: float = 0.1          # eV — Lorentzian broadening
    n_extra_bands: int = 8    # conduction bands added above the occupied set
    velocity: str = "full"    # full (∂H/∂k incl. nonlocal [V_nl,r]) | local (kinetic only)
    local_fields: bool = False  # RPA local-field effects via the Dyson ε=1−vχ₀
    scissor: float = 0.0      # eV — rigid conduction-band shift to correct the DFT gap
    dk: float = 1.0e-3        # Å⁻¹ — finite-difference step for the nonlocal velocity

    def __post_init__(self):
        if self.omega_max <= 0.0:
            raise InputError(f"optics.omega_max must be > 0, got {self.omega_max}")
        if self.n_omega < 2:
            raise InputError(f"optics.n_omega must be >= 2, got {self.n_omega}")
        if self.eta <= 0.0:
            raise InputError(f"optics.eta must be > 0, got {self.eta}")
        if self.n_extra_bands < 1:
            raise InputError(
                f"optics.n_extra_bands must be >= 1, got {self.n_extra_bands}")
        if self.velocity not in ("full", "local"):
            raise InputError(
                f"optics.velocity must be 'full' or 'local', got {self.velocity!r}")
        if self.dk <= 0.0:
            raise InputError(f"optics.dk must be > 0, got {self.dk}")


@dataclass(frozen=True)
class PhononParams:
    """Supercell finite-displacement phonons: dispersion along a q-path + DOS."""

    supercell: tuple[int, int, int] = (2, 2, 2)   # diagonal (n1,n2,n3) supercell for the FD FC
    displacement: float = 0.01     # atomic displacement h [Å] for the central FD
    path: str = ""                 # ASE bandpath string (e.g. "GXWKGL"); "" = default
    npoints: int = 120             # q-points along the dispersion path
    dos_mesh: tuple[int, int, int] = (8, 8, 8)    # MP q-mesh for the phonon DOS ((0,0,0) = skip)
    dos_width: float = 6.0         # Gaussian broadening for the DOS [cm⁻¹]
    # Temperatures [K] at which harmonic thermodynamics (F, U, Cv, S) are
    # tabulated from the DOS. Emitted only when a DOS is computed (dos_mesh > 0).
    thermo_temperatures: tuple[float, ...] = (
        0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)
    use_displacement_symmetry: bool = False  # run only point-group-irreducible displacements
    # SeedPool: run the 6·N_prim displacement SCFs across this many worker
    # processes (each warm-started from the shared undisplaced reference). 1 =
    # serial (default). forward-only — a differentiable phonon run must stay at 1.
    n_workers: int = 1

    # Infrared spectrum: Born effective charges Z* (Berry-phase finite differences
    # on the PRIMITIVE cell) → per-mode IR intensities on the Γ modes, plus a
    # Lorentzian-broadened spectrum. Norm-conserving, insulators only. Emits an
    # "ir" block from run_phonons. The extra 1 + 6·N_prim primitive-cell SCFs run
    # on a full (symmetry-unreduced) k-mesh: ir_kmesh, or inp.kpoints.mesh if 0.
    ir: bool = False
    ir_kmesh: tuple[int, int, int] = (0, 0, 0)  # Berry-phase mesh; (0,0,0) = kpoints.mesh
    born_displacement: float = 2.0e-3  # Z* finite-difference step [Å]
    ir_broadening: float = 8.0         # Lorentzian HWHM for the IR spectrum [cm⁻¹]
    # LO–TO splitting: add the q̂→0 non-analytic term to the Γ dynamical matrix
    # (needs ε∞). When on, ε∞ AND the Born charges are taken from the clamped-ion
    # E-field DFPT dielectric response (postscf.dielectric.dielectric_born, one
    # reference primitive SCF) rather than the Berry-phase FD, so ε∞/Z*/LO are
    # mutually self-consistent. Norm-conserving insulators only.
    lo_to: bool = False
    # q̂ direction for the LO–TO split; (0,0,0) = isotropic (powder) average.
    ir_qdir: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "supercell", tuple(int(n) for n in self.supercell))
        object.__setattr__(self, "dos_mesh", tuple(int(n) for n in self.dos_mesh))
        object.__setattr__(
            self, "thermo_temperatures",
            tuple(float(t) for t in self.thermo_temperatures))
        object.__setattr__(self, "ir_kmesh", tuple(int(n) for n in self.ir_kmesh))
        object.__setattr__(self, "ir_qdir", tuple(float(x) for x in self.ir_qdir))
        if len(self.ir_qdir) != 3:
            raise InputError(
                f"phonons.ir_qdir must be 3 floats, got {self.ir_qdir}")
        if len(self.ir_kmesh) != 3 or min(self.ir_kmesh) < 0:
            raise InputError(
                f"phonons.ir_kmesh must be 3 non-negative ints, got {self.ir_kmesh}")
        if not 0.0 < self.born_displacement < 0.5:
            raise InputError(
                "phonons.born_displacement must be in (0, 0.5) Å, got "
                f"{self.born_displacement}")
        if self.ir_broadening <= 0.0:
            raise InputError(
                f"phonons.ir_broadening must be > 0, got {self.ir_broadening}")
        if len(self.supercell) != 3 or min(self.supercell) < 1:
            raise InputError(
                f"phonons.supercell must be 3 positive ints, got {self.supercell}")
        if not 0.0 < self.displacement < 0.5:
            raise InputError(
                f"phonons.displacement must be in (0, 0.5) Å, got {self.displacement}")
        if len(self.dos_mesh) != 3 or min(self.dos_mesh) < 0:
            raise InputError(
                f"phonons.dos_mesh must be 3 non-negative ints, got {self.dos_mesh}")
        if any(t < 0.0 for t in self.thermo_temperatures):
            raise InputError(
                "phonons.thermo_temperatures must be non-negative, got "
                f"{self.thermo_temperatures}")
        if self.n_workers < 1:
            raise InputError(
                f"phonons.n_workers must be >= 1, got {self.n_workers}")


@dataclass(frozen=True)
class NebParams:
    """CI-NEB transition-state search between two relaxed endpoints (task: neb).

    The top-level ``structure`` is the initial state (IS); ``final`` names the
    final-state (FS) geometry file — same cell, species and atom order. A band of
    ``n_images`` (endpoints included) is built between them by ``interpolation``,
    each interior image is driven by its own GradWave calculator (the slab stack
    — slab k-mesh, per-image density warm-start across band steps — is inherited
    from the ordinary SCF calculator), and the band is relaxed to the
    minimum-energy path by the climbing-image nudged elastic band. The reported
    barrier is ``E_a = E(TS) − E(IS)`` with the TS the highest-energy image.
    """

    # final-state structure file; resolved relative to the input file's directory
    # in the parser (stored as an absolute Path). Required for a neb run.
    final: Path | None = None
    final_format: str | None = None  # ASE format override for the FS file
    final_index: int = -1            # frame to read from a multi-frame FS file
    n_images: int = 7                # TOTAL images, the two fixed endpoints included
    spring_k: float = 0.1            # band spring constant [eV/Å²]
    climb: bool = True               # climbing image: the barrier-top image climbs
    interpolation: str = "idpp"      # idpp | linear initial band
    optimizer: str = "fire"          # fire | bfgs band optimizer
    fmax: float = 0.05               # band force convergence [eV/Å]
    max_steps: int = 200
    # SeedPool: evaluate the interior images' SCFs across this many worker
    # processes per band step (each warm-started from that image's own previous
    # checkpoint). 1 = serial (default). forward-only — the parallel path returns
    # plain forces, so a differentiable NEB must stay at 1.
    n_workers: int = 1

    def __post_init__(self):
        if self.final is not None:
            object.__setattr__(self, "final", Path(self.final))
        if self.n_images < 3:
            raise InputError(
                f"neb.n_images must be >= 3 (two endpoints + a moving image), "
                f"got {self.n_images}")
        if self.spring_k <= 0.0:
            raise InputError(f"neb.spring_k must be > 0, got {self.spring_k}")
        if self.interpolation not in ("idpp", "linear"):
            raise InputError(
                f"neb.interpolation must be 'idpp' or 'linear', got "
                f"{self.interpolation!r}")
        if self.optimizer not in ("fire", "bfgs"):
            raise InputError(
                f"neb.optimizer must be 'fire' or 'bfgs', got {self.optimizer!r}")
        if self.fmax <= 0.0:
            raise InputError(f"neb.fmax must be > 0, got {self.fmax}")
        if self.max_steps < 1:
            raise InputError(f"neb.max_steps must be >= 1, got {self.max_steps}")
        if self.n_workers < 1:
            raise InputError(f"neb.n_workers must be >= 1, got {self.n_workers}")


@dataclass(frozen=True)
class CohpParams:
    """Crystal Orbital Hamilton Population, computed alongside the PDOS.

    Mirrors ``postscf.cohp.cohp``: ``pairs`` selects 0-based atom index tuples
    (``None`` → every pair within ``rcut`` Å); ``rcut`` is the neighbour cutoff,
    ``width``/``npoints`` the energy-grid broadening. Off by default; enable it
    under ``projections.cohp``."""

    enabled: bool = False
    # ((i, j), ...) 0-based; None → all within rcut
    pairs: tuple[tuple[int, int], ...] | None = None
    rcut: float = 3.0            # Å neighbour cutoff for the default pair list
    width: float = 0.1           # eV gaussian broadening
    npoints: int = 800

    def __post_init__(self):
        if self.pairs is not None:
            try:
                pairs = tuple((int(i), int(j)) for i, j in self.pairs)
            except (TypeError, ValueError) as exc:
                raise InputError(
                    "projections.cohp.pairs must be a list of [i, j] atom "
                    "index pairs") from exc
            object.__setattr__(self, "pairs", pairs)
        if self.rcut <= 0:
            raise InputError("projections.cohp.rcut must be positive")


@dataclass(frozen=True)
class ProjectionsParams:
    enabled: bool = False
    group_by: str = "l"      # atom | l | lm | total (j | jmj for FR)
    width: float = 0.1       # gaussian broadening [eV]
    npoints: int = 800
    cohp: CohpParams = field(default_factory=CohpParams)


@dataclass(frozen=True)
class DispersionParams:
    """Opt-in Grimme D3(BJ)/D4(BJ) dispersion correction (energy+forces+stress).

    A geometric, SCF-independent pairwise correction. ``method`` selects ``'d3'``
    (default) or ``'d4'`` (charge-dependent EEQ C6, with a periodic Ewald charge
    model). ``functional`` selects the BJ damping preset (defaults to the SCF
    ``xc``); s6/s8/a1/a2 override it. The cutoffs are the real-space image radii
    for the dispersion and coordination-number sums, in Å. ``charge`` is the
    total cell charge fed to the D4 EEQ model (ignored for D3)."""

    enabled: bool = False
    method: str = "d3"             # 'd3' | 'd4'
    functional: str | None = None  # None → use the SCF xc functional
    cutoff: float = 21.2       # Å, dispersion real-space image radius (~40 a0)
    cn_cutoff: float = 10.6    # Å, coordination-number image radius (~20 a0)
    s6: float | None = None
    s8: float | None = None
    a1: float | None = None
    a2: float | None = None    # Å for the a2 BJ radius override (converted at use)
    charge: float = 0.0        # D4 EEQ total charge (ignored for D3)

    def __post_init__(self):
        if self.method.lower() not in ("d3", "d4"):
            raise InputError("dispersion.method must be 'd3' or 'd4'")
        if self.cutoff <= 0 or self.cn_cutoff <= 0:
            raise InputError("dispersion cutoffs must be positive")
        if self.cn_cutoff > self.cutoff:
            raise InputError("dispersion.cn_cutoff must not exceed dispersion.cutoff")


@dataclass(frozen=True)
class VolumetricParams:
    """Volumetric fields to export after an SCF, as .cube/.xsf for VESTA/Ovito."""

    density: bool = False        # ρ(r), the CHGCAR analog
    elf: bool = False            # electron localization function ELF(r)
    magnetization: bool = False  # |m(r)|, noncollinear/SOC runs only
    bands: tuple[tuple[int, int], ...] = ()            # (band, kpoint) pairs → PARCHG |ψ_nk(r)|²
    format: str = "cube"         # "cube" or "xsf"

    def any(self) -> bool:
        return bool(self.density or self.elf or self.magnetization or self.bands)


@dataclass(frozen=True)
class EOSParams:
    """Isotropic volume scan → 3rd-order Birch-Murnaghan fit (V0, B0, B0')."""

    # volume factors relative to the input cell; the default is the calcDelta /
    # Lejaeghere seven-point window (94–106% of V0). Needs ≥4 points to fit.
    scales: tuple[float, ...] = (0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06)
    energy: str = "free_energy"  # free_energy | total | e0 — quantity fitted vs V
    # SeedPool: evaluate the volumes across this many worker processes, each
    # warm-started from the reference volume (nearest 1.0). 1 = serial (default,
    # the exact neighbour-chained warm start). >1 trades the neighbour chain for
    # a shared seed, so E(V) matches the serial fit to SCF tolerance, not
    # bit-for-bit. forward-only — a differentiable EOS must stay at 1.
    n_workers: int = 1

    def __post_init__(self):
        # coerce a YAML list to a tuple (frozen dataclass hashability) and
        # validate at parse time rather than deep in the driver
        object.__setattr__(self, "scales", tuple(float(s) for s in self.scales))
        if len(self.scales) < 4:
            raise InputError(
                f"eos.scales needs >=4 volume factors for a Birch-Murnaghan "
                f"fit, got {len(self.scales)}")
        if self.energy not in ("free_energy", "total", "e0"):
            raise InputError(
                f"unknown eos.energy {self.energy!r} (free_energy | total | e0)")
        if self.n_workers < 1:
            raise InputError(
                f"eos.n_workers must be >= 1, got {self.n_workers}")


@dataclass(frozen=True)
class ElasticParams:
    """Elastic constants: FD of the analytic stress over the six Voigt strains
    → the 6×6 stiffness C (and Voigt–Reuss–Hill moduli).

    ``mode`` selects the tensor. ``"clamped"`` (default) strains the cell with
    fractional coordinates held fixed — exact where symmetry forbids internal
    displacement, a large overestimate for soft shear constants elsewhere.
    ``"relaxed"`` re-relaxes the internal coordinates at every strained cell
    (fixed-cell BFGS to ``fmax``) before differentiating the stress, giving the
    relaxed-ion tensor that compares to experiment."""

    strain: float = 0.005  # Voigt strain magnitude h for the central difference
    mode: str = "clamped"  # clamped | relaxed (relaxed-ion: per-strain ionic relax)
    fmax: float = 0.01     # per-strain ionic relax force gate [eV/Å] (relaxed mode)
    max_steps: int = 100   # per-strain ionic relax step cap (relaxed mode)
    use_strain_symmetry: bool = False  # run only Laue-point-group-irreducible strains
    # SeedPool: run the 12 strain SCFs across this many worker processes, each
    # warm-started from the shared unstrained reference. 1 = serial (default).
    # Parallelizes the CLAMPED-ion path only; relaxed-ion stays serial (its
    # per-strain BFGS relax is a nested optimization, not a single forward SCF).
    # forward-only — a differentiable elastic run must stay at 1.
    n_workers: int = 1

    def __post_init__(self):
        if not 0.0 < self.strain < 0.1:
            raise InputError(
                f"elastic.strain must be in (0, 0.1), got {self.strain}")
        if self.mode not in ("clamped", "relaxed"):
            raise InputError(
                f"elastic.mode must be 'clamped' or 'relaxed', got {self.mode!r}")
        if not self.fmax > 0.0:
            raise InputError(f"elastic.fmax must be > 0, got {self.fmax}")
        if self.max_steps < 1:
            raise InputError(
                f"elastic.max_steps must be >= 1, got {self.max_steps}")
        if self.n_workers < 1:
            raise InputError(
                f"elastic.n_workers must be >= 1, got {self.n_workers}")


@dataclass(frozen=True)
class MagnetismParams:
    exchange: bool = True      # extract J/D from the torque (adds ~3 constrained SCFs)
    lam: float = 8.0           # constraint penalty strength [eV/μB²]
    delta: float = 0.08        # moment-tilt step for the torque derivative [rad]
    seed_scale: float = 1.5    # high-spin seed for the reference SCF (multi-stable)
    ref_atom: int = 0          # atom whose moment is tilted for the exchange scan


@dataclass(frozen=True)
class HybridParams:
    """Self-consistent PBE0-form / screened hybrid exchange (norm-conserving,
    nspin=1). Enabled by ``xc: pbe0`` (full PBE0, mode ``full``) or ``xc: hse``
    (screened, mode ``short_range``); the ``hybrid`` block overrides the mixing
    fraction ``alpha`` and the range-separation length ``omega`` [Å⁻¹]. Dispatches
    through ``postscf.hybrid.hybrid_scf``. ``name`` records the requested
    functional for the report; ``base`` is the semilocal functional the exact
    exchange is mixed into (PBE)."""

    enabled: bool = False
    name: str = "pbe0"         # requested hybrid label (display only)
    mode: str = "full"         # full (PBE0) | short_range (HSE) | long_range
    alpha: float = 0.25        # exact-exchange mixing fraction
    omega: float = 0.2         # range-separation length [Å⁻¹] (screened modes)
    base: str = "pbe"          # semilocal base functional

    def __post_init__(self):
        if self.mode not in ("full", "short_range", "long_range"):
            raise InputError(
                f"hybrid.mode must be full | short_range | long_range, "
                f"got {self.mode!r}")
        if not 0.0 <= self.alpha <= 1.0:
            raise InputError("hybrid.alpha must be in [0, 1]")
        if self.mode != "full" and self.omega <= 0.0:
            raise InputError(
                "hybrid.omega (range-separation length [Å⁻¹]) must be positive "
                "for a screened hybrid")


@dataclass(frozen=True)
class HubbardManifoldSpec:
    """One DFT+U (Dudarev) correction: U (and Hund J) on the ``l`` shell of every
    atom of element ``species``. ``l`` picks the correlated manifold (2 = d,
    3 = f, 1 = p, 0 = s); the pseudopotential must carry that atomic orbital
    (``PP_PSWFC``). Energies in eV. ``U_eff = U − J`` is what enters Dudarev."""

    species: str            # element symbol (must appear in the structure)
    l: int                  # angular momentum of the correlated shell
    u: float                # Hubbard U [eV]
    j: float = 0.0          # Hund J [eV]; enters Dudarev only as U_eff = U − J

    def __post_init__(self):
        if self.l not in (0, 1, 2, 3):
            raise InputError(
                f"hubbard.l must be 0 (s) | 1 (p) | 2 (d) | 3 (f), got {self.l}")


@dataclass(frozen=True)
class HubbardParams:
    """Rotationally-invariant DFT+U (Dudarev) applied in the SCF. Off by default;
    a ``hubbard`` block is a list of per-species manifolds. The +U occupation
    correction threads through the norm-conserving (collinear AND noncollinear/
    spin-orbit — the 2×2 spin-block occupation matrix) and USPP/PAW-collinear
    SCF; its force/stress terms are added on the collinear paths (USPP/PAW +U
    stress stays gated for the noncollinear USPP/PAW SCF, which does not yet
    have +U wired at all — see scf.uspp_noncollinear.scf_uspp_noncollinear).

    ``occ_mix`` (β, default 1.0 = today's raw one-step lag) damps the occupation
    matrix across SCF iterations; ``u_ramp_iters`` (default 0 = off) linearly
    ramps U_eff over the first N iterations. Both are convergence aids for
    large-U +U on metallic systems (see docs/manual/hubbard-u.md); set them by
    giving the ``hubbard`` block as a mapping ``{manifolds: [...], occ_mix: ...,
    u_ramp_iters: ...}`` instead of a bare list."""

    enabled: bool = False
    manifolds: tuple[HubbardManifoldSpec, ...] = ()
    occ_mix: float = 1.0
    u_ramp_iters: int = 0


@dataclass(frozen=True)
class FlapwParams:
    """All-electron FLAPW muffin-tin system controls (task: flapw, and the EFG
    NMR task). FLAPW is all-electron: the plane-wave ``pseudopotentials``, the
    top-level ``ecut`` and the ``smearing`` block do NOT apply — the muffin-tin
    stack carries its own interstitial plane-wave cutoff (``ecut`` here, in FLAPW
    units, distinct from the PW ecut in eV), per-species muffin-tin radii, and
    its own Fermi width. Drives ``flapw.crystal_scf_multi``; the cell comes from
    the top-level ``structure`` (Å, converted to Bohr) and the k-mesh from
    ``kpoints``.

    ``radii`` maps each species to its muffin-tin radius R_MT (Å) and must cover
    every element. ``los``/``val_e``/``core``/``el_override`` are the LAPW+LO
    basis overrides passed straight through: ``los = {species: [[l, spec], ...]}``
    where ``spec`` is an orbital label (``"3p"``), an energy [eV], or a mapping
    ``{"e": <label-or-eV>, "confine": false}`` (an unconfined HELO); ``val_e =
    {species: n}`` raises the valence count when an LO carries semicore, ``core =
    {species: [[l, n, occ], ...]}`` overrides the frozen core, and
    ``el_override = {species: {l: spec}}`` moves a linearization energy.

    ``efg_anion_basis = [species, ...]`` opts the named anion species into the
    validated EFG anion recipe (an unconfined l=1 HELO + the l=0 2s→2p semicore LO;
    see ``flapw.basis.efg_anion_basis``) — the accuracy recipe for the on-site
    aspherical EFG density and biaxial frame. It is merged as the base; any explicit
    ``los``/``el_override`` for the same species overrides it. It is opt-in per named
    species (anion role is compound-dependent), not applied by element."""

    radii: dict[str, float] = field(default_factory=dict)  # {species: R_MT} in Å
    ecut: float = 200.0        # interstitial plane-wave cutoff (FLAPW units)
    lmax: int = 2              # augmentation angular-momentum cutoff
    fullpot: bool = False      # self-consistent non-spherical (full) potential
    fullpot_lmax: int = 2      # non-spherical potential L cutoff (odd L included)
    iters: int = 40            # SCF iteration cap
    tol: float = 3.0e-3        # SCF residual convergence gate
    smearing: float = 0.0      # Fermi-Dirac width [eV]; 0 = exact insulator fill
    kerker: float | None = None  # interstitial Kerker screen [Å⁻¹] (fullpot aid)
    kworkers: int = 1          # k-point process-pool size
    los: dict[str, Any] | None = None      # per-species LAPW+LO specs
    val_e: dict[str, int] | None = None    # per-species valence-electron count
    core: dict[str, Any] | None = None     # per-species frozen-core override
    el_override: dict[str, Any] | None = None  # per-species linearization energy
    efg_anion_basis: list[str] | None = None   # anion species → apply the validated EFG
    #                                            anion recipe (l=1 HELO + l=0 2s→2p);
    #                                            explicit los/el_override override it per species

    def __post_init__(self):
        if self.ecut <= 0.0:
            raise InputError(f"flapw.ecut must be > 0, got {self.ecut}")
        if self.lmax < 0 or self.fullpot_lmax < 0:
            raise InputError("flapw.lmax and flapw.fullpot_lmax must be >= 0")
        if self.iters < 1:
            raise InputError(f"flapw.iters must be >= 1, got {self.iters}")
        if self.smearing < 0.0:
            raise InputError(f"flapw.smearing must be >= 0 eV, got {self.smearing}")
        if self.kworkers < 1:
            raise InputError(f"flapw.kworkers must be >= 1, got {self.kworkers}")
        if any(r <= 0.0 for r in self.radii.values()):
            raise InputError(
                "flapw.radii muffin-tin radii must be positive (Å)")
        if self.efg_anion_basis is not None:
            if len(set(self.efg_anion_basis)) != len(self.efg_anion_basis):
                raise InputError("flapw.efg_anion_basis has duplicate species")
            unknown = set(self.efg_anion_basis) - set(self.radii)
            if unknown:
                raise InputError(
                    f"flapw.efg_anion_basis names species with no flapw.radii: "
                    f"{sorted(unknown)}")


@dataclass(frozen=True)
class NmrSpectrumParams:
    """Powder-lineshape synthesis for the shielding NMR task (``postscf.nmr_spectrum``).

    Off by default (``enabled=False``), so a shielding run stays byte-identical
    unless a spectrum is asked for. When enabled the driver assembles one
    :class:`gradwave.postscf.nmr_spectrum.NMRSite` per referenced site — the
    chemical shift δ_iso (from ``nmr.sigma_ref``; the spectrum needs a reference),
    the CSA (δ_aniso, η_csa) from the shielding tensor, and — for a quadrupolar
    half-integer nucleus — C_Q/η_Q/spin from the ``nmr.efg`` block — then calls the
    powder-average synthesis and emits ``(ppm_axis, intensity)``.

    ``mode`` selects the lineshape family: ``'static'`` (a static powder pattern)
    or ``'mas'`` (magic-angle spinning at ``spin_rate_hz``). A spin-½ observed
    nucleus uses the CSA lineshape (the shielding alone sets it); a quadrupolar
    half-integer nucleus uses the second-order central-transition lineshape.
    ``larmor_mhz`` is the observed-nucleus Larmor frequency (a single value, or a
    species/isotope → MHz map) — required for MAS and quadrupolar lineshapes,
    unused for a spin-½ static CSA pattern. ``broadening_ppm`` convolves a
    ``lineshape`` (``'gauss'`` | ``'lorentz'``) line; the powder average uses
    ``n_orientations`` orientations over ``n_points`` axis bins."""

    enabled: bool = False
    mode: str = "static"  # static | mas
    spin_rate_hz: float = 0.0  # MAS rotor frequency (mode='mas')
    # observed-nucleus Larmor frequency (MHz): a single value or a species/isotope map
    larmor_mhz: float | dict[str, float] | None = None
    broadening_ppm: float = 0.0
    lineshape: str = "gauss"  # gauss | lorentz
    n_orientations: int = 2000
    n_points: int = 2048

    def __post_init__(self):
        if self.mode not in ("static", "mas"):
            raise InputError(
                f"nmr.spectrum.mode must be 'static' or 'mas', got {self.mode!r}")
        if self.lineshape not in ("gauss", "lorentz"):
            raise InputError(
                "nmr.spectrum.lineshape must be 'gauss' or 'lorentz', got "
                f"{self.lineshape!r}")
        if self.enabled and self.mode == "mas" and self.spin_rate_hz <= 0.0:
            raise InputError(
                "nmr.spectrum.spin_rate_hz must be > 0 Hz for mode='mas'")
        if self.n_orientations < 1:
            raise InputError("nmr.spectrum.n_orientations must be >= 1")
        if self.n_points < 2:
            raise InputError("nmr.spectrum.n_points must be >= 2")
        if self.broadening_ppm < 0.0:
            raise InputError("nmr.spectrum.broadening_ppm must be >= 0 ppm")


@dataclass(frozen=True)
class NmrParams:
    """NMR-observable selection (task: nmr).

    ``task='efg'`` computes the electric field gradient tensor per site through
    the all-electron FLAPW stack (needs a ``flapw`` block; ``pseudopotentials``
    are not used) and, for the selected isotopes, the quadrupolar coupling C_Q =
    2.4180·Q[barn]·V_zz[eV/Å²]. ``task='shielding'`` computes the magnetic
    shielding tensor per site through the plane-wave GIPAW route (needs
    ``pseudopotentials``, ``ecut`` and a k-mesh with ≥2 axes of length > 1).

    ``shielding_level`` selects the shielding assembly (``task='shielding'``
    only): ``'bare'`` is the smooth valence term alone (the analytic q→0 route
    ``kgeometry_nmr.sigma_shielding_dq``; a norm-conserving ground state),
    ``'gipaw'`` is the full absolute σ = σ_bare + σ_core + σ_dia_aug + σ_para_aug
    (``kgeometry_nmr.sigma_shielding_gipaw``; needs an all-PAW ground state), and
    ``'auto'`` (default) picks ``'gipaw'`` when the pseudopotentials are PAW and
    ``'bare'`` otherwise.

    ``sigma_ref`` maps a species or isotope label to a reference absolute
    shielding σ_ref (ppm); when given, each matching site also reports the
    chemical shift δ_iso = σ_ref − σ_iso. Obtain σ_ref for a species by running
    the same-level shielding on a reference solid (``api.reference_sigma_iso``).

    ``isotopes`` maps a species to the isotope whose Q gives C_Q (e.g.
    ``{"Ti": "49Ti", "O": "17O"}``); a species left unmapped reports V_zz/η only.
    None auto-selects the first tabulated isotope for each element (EFG only).

    ``efg`` (``task='shielding'`` only) also computes the plane-wave/PAW electric
    field gradient (Petrilli–Blöchl, ``postscf.efg_paw``) on the SAME ground state
    as the shielding, reporting per-site V_zz/η and — for the ``isotopes`` — C_Q:
    ``False`` off, ``True`` on (requires an all-PAW ground state), ``'auto'`` on
    only when the ground state is PAW. It feeds the quadrupolar spectrum path.

    ``spectrum`` (:class:`NmrSpectrumParams`) synthesizes a powder lineshape from
    the per-site δ_iso + CSA (+ C_Q/η_Q for quadrupolar nuclei); off by default.

    ``chunk_k`` (``task='shielding'``, PW route) streams the dense per-k
    response contexts over blocks of at most that many k-points — bit-identical
    to the eager default (None) but with an O(1)-in-nk peak memory footprint,
    at the cost of rebuilding each context once per sampled axis. The memory
    route for cells whose full-mesh contexts do not fit in RAM."""

    task: str = "efg"  # efg | shielding
    isotopes: dict[str, str] | None = None
    shielding_level: str = "auto"  # auto | bare | gipaw (task: shielding)
    # species/isotope -> reference σ (ppm) for the chemical shift δ_iso = σ_ref − σ_iso
    sigma_ref: dict[str, float] | None = None
    efg: bool | str = False  # False | True | 'auto': PW/PAW EFG within task='shielding'
    spectrum: NmrSpectrumParams = field(default_factory=NmrSpectrumParams)
    chunk_k: int | None = None  # stream dense response contexts (memory route)

    def __post_init__(self):
        if self.task not in ("efg", "shielding"):
            raise InputError(
                f"nmr.task must be 'efg' or 'shielding', got {self.task!r}")
        if self.shielding_level not in ("auto", "bare", "gipaw"):
            raise InputError(
                "nmr.shielding_level must be 'auto', 'bare' or 'gipaw', got "
                f"{self.shielding_level!r}")
        if self.efg not in (True, False, "auto"):
            raise InputError(
                f"nmr.efg must be true, false or 'auto', got {self.efg!r}")
        if self.chunk_k is not None and self.chunk_k < 1:
            raise InputError(
                f"nmr.chunk_k must be >= 1 (or null for eager), got {self.chunk_k}")


@dataclass(frozen=True)
class Input:
    atoms: Atoms
    pseudo_dir: Path
    pseudo_map: dict[str, str]
    ecut: float
    ecutrho: float | None = None  # USPP/PAW density cutoff; None → 4×ecut
    xc: str = "pbe"  # lda | pbe | r2scan (semilocal base for a hybrid xc)
    hybrid: HybridParams = field(default_factory=HybridParams)  # pbe0 | hse
    hubbard: HubbardParams = field(default_factory=HubbardParams)  # DFT+U
    kpoints: KPointsParams = field(default_factory=KPointsParams)
    smearing: SmearingParams = field(default_factory=SmearingParams)
    nbands: int | None = None
    scf: SCFParams = field(default_factory=SCFParams)
    slab: SlabParams = field(default_factory=SlabParams)  # ESM vacuum auto-sizer
    symmetry: bool = True  # IBZ reduction + density symmetrization
    nspin: int = 1  # 1 | 2 (collinear)
    noncollinear: bool = False  # spinor (non-collinear) SCF for task: scf
    nonmagnetic: bool = False  # with noncollinear: pin m⃗ ≡ 0 (spin-orbit only, keeps symmetry)
    # element -> initial moment fraction (nspin=2/NC seed)
    start_mag: dict[str, float] | None = None
    tot_magnetization: float | None = None  # fix M=N↑−N↓ (nspin=2): integer fill
    # without smearing, two-Fermi-level smeared FSM with smearing
    # scf | relax | neb | bands | optics | magnetism | eos | elastic | phonons | flapw | nmr
    task: str = "scf"
    relax: RelaxParams = field(default_factory=RelaxParams)
    neb: NebParams = field(default_factory=NebParams)  # CI-NEB transition state
    bands: BandsParams = field(default_factory=BandsParams)
    optics: OpticsParams = field(default_factory=OpticsParams)
    magnetism: MagnetismParams = field(default_factory=MagnetismParams)
    eos: EOSParams = field(default_factory=EOSParams)
    elastic: ElasticParams = field(default_factory=ElasticParams)
    phonons: PhononParams = field(default_factory=PhononParams)
    flapw: FlapwParams = field(default_factory=FlapwParams)  # all-electron FLAPW
    nmr: NmrParams = field(default_factory=NmrParams)  # EFG / shielding (task: nmr)
    projections: ProjectionsParams = field(default_factory=ProjectionsParams)
    dispersion: DispersionParams = field(default_factory=DispersionParams)
    device: str = "cpu"
    distributed: bool = False  # k-point-sharded SCF across torchrun ranks (see
    # gradwave.distributed / docs/manual/distributed.md); norm-conserving and
    # USPP/PAW collinear SCF (DFT+U included), with or without IBZ symmetry
    # reduction (the shard unit is whatever k-list the system was built with)
    verbose: bool = True  # per-iteration SCF chatter; CLI --quiet overrides
    output_dir: Path = Path("./out")
    output_checkpoint: bool = True  # write checkpoint.pt after SCF tasks
    output_wavefunctions: bool = False  # include coeffs in the checkpoint
    output_volumetric: VolumetricParams = field(default_factory=VolumetricParams)
    # post-SCF numerical-error estimates (basis set, SCF, smearing) in the
    # output — on by default; every estimate is derived from the finished
    # run (no extra SCF) and out-of-coverage runs degrade to available: false
    error_estimate: bool = True
    restart: Path | None = None  # checkpoint.pt to warm-start from (USPP/PAW)
    # selective dynamics (VASP/QE if_pos analog): a (natoms, 3) boolean mask,
    # True = that fractional axis is HELD FIXED during a relax, False = free.
    # None (the default) relaxes every atom, bit-for-bit the historical path.
    # Fixed components are zeroed before the optimizer and before the fmax gate
    # (convergence is over free components only); held atoms ride the cell in
    # fractional coordinates under a variable-cell relax (see structure.fixed in
    # docs/manual/io.md and docs/manual/geometry-optimization.md).
    fixed: np.ndarray | None = None
