"""YAML input schema → validated frozen dataclasses (Layer C).

Geometry goes through ASE, so `structure:` accepts any format ASE reads
(cif, POSCAR, xyz, ...) or an inline cell/positions/species block.
All energies in eV, lengths in Å (the package-wide convention).
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import yaml
from ase import Atoms
from ase.io import read as ase_read

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_T = TypeVar("_T", bound="DataclassInstance")


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
    # block eigensolver: "davidson" (the workhorse) | "chebyshev" (Chebyshev-
    # filtered subspace iteration). Norm-conserving only; the USPP/PAW generalized
    # S-metric problem rejects a non-davidson choice (see api / calculator).
    eigensolver: str = "davidson"  # davidson | chebyshev
    # electrostatic boundary: "periodic" (default 3D-periodic) | "open_z"
    # (open-boundary / ESM in the c direction — slab geometry, c ⊥ a,b). open_z
    # removes the vacuum-image dipole field so asymmetric surfaces need no dipole
    # correction and surface energetics are box-independent (core/energies/esm.py).
    boundary: str = "periodic"
    # magnetic-channel SCF controls for the noncollinear/spinor path only
    magnetic: MagneticParams = field(default_factory=MagneticParams)

    def __post_init__(self):
        if self.convergence not in ("density", "energy"):
            raise InputError(
                "scf.convergence must be 'density' or 'energy', got "
                f"{self.convergence!r}")
        if self.eigensolver not in ("davidson", "chebyshev"):
            raise InputError(
                "scf.eigensolver must be 'davidson' or 'chebyshev', got "
                f"{self.eigensolver!r}")
        if self.boundary not in ("periodic", "open_z"):
            raise InputError(
                "scf.boundary must be 'periodic' or 'open_z', got "
                f"{self.boundary!r}")


@dataclass(frozen=True)
class SmearingParams:
    type: str = "none"  # none | fermi-dirac | gaussian | mp1 | cold
    width: float = 0.1  # eV


@dataclass(frozen=True)
class KPointsParams:
    mesh: tuple[int, int, int] = (1, 1, 1)
    shift: tuple[int, int, int] = (0, 0, 0)


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

    def __post_init__(self):
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


@dataclass(frozen=True)
class BandsParams:
    path: str = ""  # ASE bandpath string, e.g. "LGXUG"; empty = ASE default
    npoints: int = 120
    nbands: int | None = None
    irreps: bool = False  # label bands at special points with Mulliken symbols


@dataclass(frozen=True)
class PhononParams:
    """Supercell finite-displacement phonons: dispersion along a q-path + DOS."""

    supercell: tuple[int, int, int] = (2, 2, 2)   # diagonal (n1,n2,n3) supercell for the FD FC
    displacement: float = 0.01     # atomic displacement h [Å] for the central FD
    path: str = ""                 # ASE bandpath string (e.g. "GXWKGL"); "" = default
    npoints: int = 120             # q-points along the dispersion path
    dos_mesh: tuple[int, int, int] = (8, 8, 8)    # MP q-mesh for the phonon DOS ((0,0,0) = skip)
    dos_width: float = 6.0         # Gaussian broadening for the DOS [cm⁻¹]

    def __post_init__(self):
        object.__setattr__(self, "supercell", tuple(int(n) for n in self.supercell))
        object.__setattr__(self, "dos_mesh", tuple(int(n) for n in self.dos_mesh))
        if len(self.supercell) != 3 or min(self.supercell) < 1:
            raise InputError(
                f"phonons.supercell must be 3 positive ints, got {self.supercell}")
        if not 0.0 < self.displacement < 0.5:
            raise InputError(
                f"phonons.displacement must be in (0, 0.5) Å, got {self.displacement}")
        if len(self.dos_mesh) != 3 or min(self.dos_mesh) < 0:
            raise InputError(
                f"phonons.dos_mesh must be 3 non-negative ints, got {self.dos_mesh}")


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
    symmetry: bool = True  # IBZ reduction + density symmetrization
    nspin: int = 1  # 1 | 2 (collinear)
    noncollinear: bool = False  # spinor (non-collinear) SCF for task: scf
    nonmagnetic: bool = False  # with noncollinear: pin m⃗ ≡ 0 (spin-orbit only, keeps symmetry)
    # element -> initial moment fraction (nspin=2/NC seed)
    start_mag: dict[str, float] | None = None
    tot_magnetization: float | None = None  # fix M=N↑−N↓ (nspin=2, no smearing)
    task: str = "scf"  # scf | relax | bands | magnetism | eos | elastic | phonons
    relax: RelaxParams = field(default_factory=RelaxParams)
    bands: BandsParams = field(default_factory=BandsParams)
    magnetism: MagnetismParams = field(default_factory=MagnetismParams)
    eos: EOSParams = field(default_factory=EOSParams)
    elastic: ElasticParams = field(default_factory=ElasticParams)
    phonons: PhononParams = field(default_factory=PhononParams)
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


def _check_keys(label: str, got: object, allowed: Iterable[str]) -> None:
    """Reject unknown keys in a mapping with a did-you-mean hint, so a typo
    like `optimzer:` fails loudly at parse time instead of being silently
    dropped (a dropped key means the default is used and the result is quietly
    wrong)."""
    if not isinstance(got, dict):
        raise InputError(f"{label} must be a mapping, got {type(got).__name__}")
    allowed = set(allowed)
    unknown = [k for k in got if k not in allowed]
    if not unknown:
        return
    parts = []
    for k in unknown:
        near = difflib.get_close_matches(str(k), [str(a) for a in allowed], n=1)
        parts.append(f"{k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    raise InputError(
        f"unknown key(s) in {label}: {', '.join(parts)}. "
        f"valid keys: {', '.join(sorted(str(a) for a in allowed))}")


def _build(cls: type[_T], raw: dict[str, Any], label: str) -> _T:
    """Construct a frozen params dataclass from a mapping, rejecting unknown
    keys first so `RelaxParams(**{'optimzer': ...})` reports the typo by name
    rather than raising a bare TypeError from the constructor."""
    _check_keys(label, raw, {f.name for f in dataclasses.fields(cls)})
    return cls(**raw)


def _build_volumetric(raw: bool | dict[str, Any]) -> VolumetricParams:
    """Parse the `output.volumetric` block. `true` is shorthand for the density
    alone; a mapping selects fields and the file format."""
    if isinstance(raw, bool):
        return VolumetricParams(density=raw)
    _check_keys("output.volumetric", raw,
                {"density", "elf", "magnetization", "bands", "format"})
    fmt = str(raw.get("format", "cube"))
    if fmt not in ("cube", "xsf", "chgcar"):
        raise InputError(
            f"output.volumetric.format must be 'cube', 'xsf' or 'chgcar', got {fmt!r}")
    try:
        bands = tuple((int(b), int(k)) for b, k in raw.get("bands", ()))
    except (TypeError, ValueError) as exc:
        raise InputError(
            "output.volumetric.bands must be a list of [band, kpoint] pairs") from exc
    return VolumetricParams(
        density=bool(raw.get("density", False)),
        elf=bool(raw.get("elf", False)),
        magnetization=bool(raw.get("magnetization", False)),
        bands=bands,
        format=fmt,
    )


def _read_atoms(path: Path, fmt: str | None = None, index: int = -1) -> Atoms:
    """Read a geometry through ASE and enforce the plane-wave prerequisites.

    ASE guesses the format from the extension/content; `fmt` overrides that
    when the guess misfires. Multi-image files (trajectories, multi-frame xyz)
    default to the last frame (`index=-1`) rather than silently — pass a
    `structure.index` to choose. A structure with no 3D cell cannot be run by a
    plane-wave code, so that fails here with a clear message rather than deep in
    grid construction."""
    try:
        atoms = ase_read(str(path), format=fmt, index=index)
    except FileNotFoundError:
        raise FileNotFoundError(f"structure file not found: {path}") from None
    except Exception as e:  # ASE raises a grab-bag of parse errors
        hint = "" if fmt else " (try setting structure.format)"
        raise InputError(f"could not read structure {path}: {e}{hint}") from None
    if isinstance(atoms, list):
        raise InputError(
            f"structure.index {index!r} selected {len(atoms)} frames; "
            f"give a single integer index (e.g. 0 or -1)")
    if atoms.cell.rank < 3:
        raise InputError(
            f"structure {path} has no 3D cell (cell rank {atoms.cell.rank}); "
            f"plane-wave DFT requires a periodic cell. If this is a molecule, "
            f"put it in a box (e.g. a POSCAR/cif with a lattice).")
    atoms.pbc = True
    return atoms


def _normalize_kerker(value: object) -> str | bool:
    """MixingParams.kerker accepts "auto", a bool, or the on/off/true/false
    string spellings (a bare `kerker: off` is already a YAML bool); anything
    else is a user error rather than a silent truthy string."""
    if value == "auto":
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "auto":
            return "auto"
        if s in ("on", "true"):
            return True
        if s in ("off", "false"):
            return False
    raise InputError(
        f"invalid mixing.kerker {value!r} (auto | on | off | true | false)")


def _parse_fixed(raw_fixed: Any, natoms: int) -> np.ndarray:
    """Parse ``structure.fixed`` into a ``(natoms, 3)`` boolean selective-dynamics
    mask where ``True`` means that fractional axis is held fixed in a relax.

    Two spellings, validated against the atom count at parse time:

      fixed: [0, 3, 4]                        # 0-based indices, atoms fully fixed
      fixed: [[true, true, true], [false, false, false], ...]  # per-atom [x,y,z]

    The index form fixes all three axes of each listed atom; the per-atom form
    needs one ``[x, y, z]`` boolean row per atom (length == natoms). A YAML bool
    is a Python ``int`` subclass, so the index form rejects booleans to keep the
    two spellings distinct.
    """
    if not isinstance(raw_fixed, (list, tuple)) or len(raw_fixed) == 0:
        raise InputError(
            "structure.fixed must be a non-empty list of atom indices or a "
            "per-atom list of [x, y, z] booleans")
    mask = np.zeros((natoms, 3), dtype=bool)
    if all(isinstance(e, (list, tuple)) for e in raw_fixed):
        # per-atom [x, y, z] boolean rows
        if len(raw_fixed) != natoms:
            raise InputError(
                f"structure.fixed per-atom mask has {len(raw_fixed)} rows but "
                f"the structure has {natoms} atoms")
        for i, row in enumerate(raw_fixed):
            if len(row) != 3:
                raise InputError(
                    f"structure.fixed[{i}] must be 3 booleans [x, y, z], got "
                    f"{len(row)}")
            for j, v in enumerate(row):
                if not isinstance(v, bool):
                    raise InputError(
                        f"structure.fixed[{i}][{j}] must be a boolean "
                        f"(true/false), got {v!r}")
                mask[i, j] = v
        if not mask.any():
            raise InputError(
                "structure.fixed pins no axes (every entry false); omit it to "
                "relax every atom")
        return mask
    if all(isinstance(e, int) and not isinstance(e, bool) for e in raw_fixed):
        # flat list of 0-based atom indices, fully fixed
        idx = list(raw_fixed)
        for i in idx:
            if not 0 <= i < natoms:
                raise InputError(
                    f"structure.fixed index {i} out of range for {natoms} atoms "
                    f"(valid 0..{natoms - 1})")
        if len(set(idx)) != len(idx):
            raise InputError(f"structure.fixed has duplicate indices: {idx}")
        mask[idx, :] = True
        return mask
    raise InputError(
        "structure.fixed must be either a flat list of integer atom indices or "
        "a per-atom list of [x, y, z] boolean rows, not a mix of the two")


def _load_structure(
    spec: str | dict[str, Any], base: Path
) -> tuple[Atoms, np.ndarray | None]:
    """Three spellings, all reaching the same Atoms (plus an optional
    selective-dynamics mask parsed from ``fixed`` in either mapping form):

      structure: geometry.cif                     # bare filename (any ASE format)
      structure: {file: t.xyz, format: extxyz, index: 0}   # file + read controls
      structure: {cell: ..., positions: ..., species: ...} # inline block

    Returns ``(atoms, fixed)`` where ``fixed`` is the ``(natoms, 3)`` boolean
    mask or ``None`` when no ``fixed`` key is given (the bare-filename form
    cannot carry one — use a mapping to fix atoms with a file geometry).
    """
    if isinstance(spec, str):
        return _read_atoms(base / spec), None
    if not isinstance(spec, dict):
        raise InputError(
            f"structure must be a filename or a mapping, got {type(spec).__name__}")
    if "file" in spec:
        _check_keys("structure", spec, {"file", "format", "index", "fixed"})
        atoms = _read_atoms(base / spec["file"], fmt=spec.get("format"),
                            index=spec.get("index", -1))
    else:
        _check_keys("structure", spec, {"cell", "positions", "species", "fixed"})
        for req in ("cell", "positions", "species"):
            if req not in spec:
                raise InputError(f"inline structure is missing required key {req!r}")
        cell = np.asarray(spec["cell"], dtype=float)
        posblock = spec["positions"]
        _check_keys("structure.positions", posblock, {"cart", "frac"})
        species = spec["species"]
        if "frac" in posblock:
            atoms = Atoms(species, scaled_positions=posblock["frac"], cell=cell,
                          pbc=True)
        elif "cart" in posblock:
            atoms = Atoms(species, positions=posblock["cart"], cell=cell, pbc=True)
        else:
            raise InputError("structure.positions needs a 'cart' or 'frac' block")
    fixed = None if "fixed" not in spec else _parse_fixed(spec["fixed"], len(atoms))
    return atoms, fixed


# Every top-level key the schema understands; anything else is a typo. Kept
# beside the Input fields it feeds so the two do not drift.
_ALLOWED_TOP = {
    "structure", "pseudopotentials", "ecut", "ecutrho", "xc", "hybrid", "hubbard",
    "kpoints", "smearing", "nbands", "symmetry", "nspin", "noncollinear",
    "nonmagnetic", "start_mag", "tot_magnetization",
    "scf", "task", "relax", "bands", "magnetism", "eos", "elastic", "phonons",
    "projections", "dispersion", "device", "distributed",
    "verbose", "output", "error_estimate", "restart",
}


def load_input(path: str | Path) -> Input:
    """Parse a YAML input into the frozen `Input` schema. Every `InputError`
    is re-raised with the file path prepended so the message names the file the
    user edited."""
    path = Path(path)
    try:
        return _load_input(path)
    except InputError as e:
        raise InputError(f"{path}: {e}") from None


def _resolve_pseudopotentials(
    pp: dict[str, Any], base: Path, symbols: Iterable[str]
) -> tuple[Path, dict[str, str]]:
    """Validate the pseudopotentials block, resolve its directory and per-element
    map, and check every element in ``symbols`` has an existing UPF file."""
    _check_keys("pseudopotentials", pp, {"dir", "map"})
    for req in ("dir", "map"):
        if req not in pp:
            raise InputError(f"pseudopotentials is missing required key {req!r}")
    pseudo_dir = (base / pp["dir"]).resolve()
    pseudo_map = dict(pp["map"])
    for sym in set(symbols):
        if sym not in pseudo_map:
            raise InputError(f"no pseudopotential mapped for element {sym}")
        resolved = pseudo_dir / pseudo_map[sym]
        if not resolved.exists():
            raise InputError(
                f"pseudopotential for element {sym!r} not found at {resolved} — "
                f"check pseudopotentials.dir and pseudopotentials.map")
    return pseudo_dir, pseudo_map


def _resolve_symmetry(raw: dict[str, Any], task: str) -> tuple[bool, bool, bool]:
    """Resolve (noncollinear, nonmagnetic, symmetry) from the raw input.

    A magnetic spinor SCF (a magnetic noncollinear run, and the magnetism
    task's constrained tilt scans) cannot use IBZ symmetry reduction: time
    reversal and the space group act on the moment vector, so the driver
    rejects a symmetrized density. Default symmetry off for these modes and
    reject an explicit ``symmetry: true`` here, where the message can point at
    the fix, rather than letting it surface as a ValueError deep in the SCF.
    A spin-orbit-only run (nonmagnetic: true) pins m⃗ ≡ 0, so Kramers keeps
    the full crystal symmetry and it behaves like a plain SCF for symmetry.
    """
    noncollinear = bool(raw.get("noncollinear", False))
    nonmagnetic = bool(raw.get("nonmagnetic", False))
    if nonmagnetic and not noncollinear:
        raise InputError(
            "nonmagnetic requires noncollinear: true — it pins the spinor "
            "moment to zero for a spin-orbit-only run")
    magnetic_spinor = (noncollinear and not nonmagnetic) or task == "magnetism"
    sym_raw = raw.get("symmetry")
    if magnetic_spinor:
        if sym_raw is True:
            mode = "magnetism" if task == "magnetism" else "noncollinear"
            raise InputError(
                f"symmetry: true is invalid for a {mode} run — time reversal "
                f"and the space group act on the moment vector, so IBZ "
                f"reduction is rejected. Set symmetry: false (the default for "
                f"these modes), or for spin-orbit without magnetism add "
                f"nonmagnetic: true, which keeps symmetry.")
        symmetry = False
    else:
        symmetry = True if sym_raw is None else bool(sym_raw)
    return noncollinear, nonmagnetic, symmetry


def _validate_mixing(mix_raw: dict[str, Any]) -> None:
    """Validate the `scf.mixing` block in place: reject unknown keys and unknown
    schemes, and normalize the `kerker` shorthand. Mutates `mix_raw['kerker']`."""
    _check_keys("scf.mixing", mix_raw,
                {f.name for f in dataclasses.fields(MixingParams)})
    mix_scheme = str(mix_raw.get("scheme", "auto"))
    if mix_scheme not in ("auto", "pulay", "broyden", "johnson"):
        raise InputError(f"unknown mixing scheme {mix_scheme!r}")
    if "precond" in mix_raw:
        precond = str(mix_raw["precond"])
        if precond not in ("kerker", "local_tf"):
            raise InputError(
                f"unknown mixing precond {precond!r} (kerker | local_tf)")
    if "kerker" in mix_raw:
        mix_raw["kerker"] = _normalize_kerker(mix_raw["kerker"])


def _build_cohp(cohp_raw: bool | dict[str, Any]) -> CohpParams:
    """Parse the `projections.cohp` sub-block. `true`/`false` is the enabled
    shorthand; a mapping selects the atom pairs, cutoff and broadening."""
    if isinstance(cohp_raw, bool):
        return CohpParams(enabled=cohp_raw)
    _check_keys("projections.cohp", cohp_raw,
                {"enabled", "pairs", "rcut", "width", "npoints"})
    # pairs is passed through raw; CohpParams.__post_init__ coerces and validates
    # it (so a malformed pair reports the InputError, not a bare TypeError here).
    return CohpParams(
        enabled=bool(cohp_raw.get("enabled", True)),
        pairs=cohp_raw.get("pairs"),
        rcut=float(cohp_raw.get("rcut", 3.0)),
        width=float(cohp_raw.get("width", 0.1)),
        npoints=int(cohp_raw.get("npoints", 800)),
    )


def _build_projections(proj_raw: bool | dict[str, Any]) -> ProjectionsParams:
    """Parse the `projections` block. `true`/`false` is the enabled shorthand;
    a mapping selects the grouping and broadening, plus an optional `cohp`
    sub-block computed alongside the PDOS."""
    if isinstance(proj_raw, bool):
        return ProjectionsParams(enabled=proj_raw)
    _check_keys("projections", proj_raw,
                {"enabled", "group_by", "width", "npoints", "cohp"})
    return ProjectionsParams(
        enabled=bool(proj_raw.get("enabled", True)),
        group_by=str(proj_raw.get("group_by", "l")),
        width=float(proj_raw.get("width", 0.1)),
        npoints=int(proj_raw.get("npoints", 800)),
        cohp=_build_cohp(proj_raw.get("cohp", False)),
    )


def _build_dispersion(disp_raw: bool | dict[str, Any]) -> DispersionParams:
    """Parse the `dispersion` block. `true`/`false` is the enabled shorthand
    (BJ damping then taken from the SCF functional); a mapping overrides the
    functional, cutoffs, and the four damping constants."""
    if isinstance(disp_raw, bool):
        return DispersionParams(enabled=disp_raw)
    _check_keys("dispersion", disp_raw,
                {"enabled", "method", "functional", "cutoff", "cn_cutoff",
                 "s6", "s8", "a1", "a2", "charge"})
    def _optf(key: str) -> float | None:
        v = disp_raw.get(key)
        return None if v is None else float(v)
    return DispersionParams(
        enabled=bool(disp_raw.get("enabled", True)),
        method=str(disp_raw.get("method", "d3")),
        functional=disp_raw.get("functional"),
        cutoff=float(disp_raw.get("cutoff", 21.2)),
        cn_cutoff=float(disp_raw.get("cn_cutoff", 10.6)),
        s6=_optf("s6"), s8=_optf("s8"), a1=_optf("a1"), a2=_optf("a2"),
        charge=float(disp_raw.get("charge", 0.0)),
    )


# xc: pbe0 / hse select a hybrid; the value is (mode, default omega). The
# semilocal base is always PBE (the hybrid machinery scales PBE exchange).
_HYBRID_PRESETS = {
    "pbe0": ("full", None),
    "hse": ("short_range", 0.2),
}


def _resolve_xc(raw: dict[str, Any]) -> tuple[str, HybridParams]:
    """Resolve (semilocal base xc, HybridParams) from the raw `xc` value and an
    optional `hybrid` override block. A plain functional (lda|pbe|r2scan) returns
    a disabled HybridParams and rejects a stray `hybrid` block; a hybrid label
    (pbe0|hse) sets the base to PBE and folds the preset with any overrides."""
    xc = str(raw.get("xc", "pbe")).lower()
    hyb_raw = raw.get("hybrid", {})
    if xc not in _HYBRID_PRESETS:
        if xc not in ("lda", "pbe", "r2scan"):
            raise InputError(
                f"unknown xc {xc!r} (lda | pbe | r2scan | pbe0 | hse)")
        if hyb_raw:
            raise InputError(
                "a hybrid block needs a hybrid functional (xc: pbe0 or hse)")
        return xc, HybridParams()
    mode, omega_def = _HYBRID_PRESETS[xc]
    _check_keys("hybrid", hyb_raw, {"alpha", "omega", "mode"})
    return "pbe", HybridParams(
        enabled=True, name=xc,
        mode=str(hyb_raw.get("mode", mode)),
        alpha=float(hyb_raw.get("alpha", 0.25)),
        omega=float(hyb_raw.get("omega", omega_def if omega_def is not None else 0.2)),
        base="pbe",
    )


def _build_hubbard(
    hub_raw: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any] | None,
    symbols: list[str],
) -> HubbardParams:
    """Parse the `hubbard` block: a list of per-species +U manifolds. Each entry
    names an element `species` present in the structure, the correlated shell
    `l`, and `u` (+ optional `j`) in eV. One manifold per species (Dudarev is a
    per-species correction); a species absent from the cell is a typo, not a
    silent no-op, so it is rejected.

    The block may also be a mapping ``{manifolds: [...], occ_mix, u_ramp_iters}``
    to carry the two convergence aids alongside the manifold list: ``occ_mix``
    (occupation-matrix damping β in (0, 1], default 1.0) and ``u_ramp_iters``
    (linear U-ramp length, non-negative integer, default 0 = off)."""
    if hub_raw is None:
        return HubbardParams()
    occ_mix, u_ramp_iters = 1.0, 0
    if isinstance(hub_raw, dict):
        # mapping form: manifolds list plus the convergence-aid knobs. Require
        # `manifolds` before validating keys so a bare single-manifold mapping
        # (no `manifolds` key) still fails with the list-form guidance.
        if "manifolds" not in hub_raw:
            raise InputError(
                "hubbard must be a list of {species, l, u[, j]} manifolds, or a "
                "mapping {manifolds: [...], occ_mix, u_ramp_iters}")
        _check_keys("hubbard", hub_raw, {"manifolds", "occ_mix", "u_ramp_iters"})
        occ_mix = float(hub_raw.get("occ_mix", 1.0))
        u_ramp_iters = int(hub_raw.get("u_ramp_iters", 0))
        if not (0.0 < occ_mix <= 1.0):
            raise InputError(
                f"hubbard.occ_mix (occupation-matrix damping) must be in "
                f"(0, 1], got {occ_mix}")
        if u_ramp_iters < 0:
            raise InputError(
                f"hubbard.u_ramp_iters must be a non-negative integer, "
                f"got {u_ramp_iters}")
        hub_raw = hub_raw["manifolds"]
    if not isinstance(hub_raw, (list, tuple)):
        raise InputError(
            "hubbard must be a list of {species, l, u[, j]} manifolds")
    present = set(symbols)
    seen: set[str] = set()
    manifolds: list[HubbardManifoldSpec] = []
    for i, entry in enumerate(hub_raw):
        _check_keys(f"hubbard[{i}]", entry, {"species", "l", "u", "j"})
        if "species" not in entry or "l" not in entry or "u" not in entry:
            raise InputError(
                f"hubbard[{i}] needs species, l and u (u in eV; l the shell)")
        sp = str(entry["species"])
        if sp not in present:
            raise InputError(
                f"hubbard[{i}]: species {sp!r} is not in the structure "
                f"(elements present: {', '.join(sorted(present))})")
        if sp in seen:
            raise InputError(
                f"hubbard: species {sp!r} appears twice (one manifold per "
                f"species — Dudarev +U applies to every atom of the element)")
        seen.add(sp)
        manifolds.append(HubbardManifoldSpec(
            species=sp, l=int(entry["l"]), u=float(entry["u"]),
            j=float(entry.get("j", 0.0))))
    return HubbardParams(enabled=bool(manifolds), manifolds=tuple(manifolds),
                         occ_mix=occ_mix, u_ramp_iters=u_ramp_iters)


def _load_input(path: Path) -> Input:
    raw_yaml: Any = yaml.safe_load(path.read_text())
    base = path.parent

    if not isinstance(raw_yaml, dict):
        raise InputError("input must be a YAML mapping of keywords")
    raw: dict[str, Any] = raw_yaml
    _check_keys("input", raw, _ALLOWED_TOP)
    for req in ("structure", "pseudopotentials", "ecut"):
        if req not in raw:
            raise InputError(f"missing required key {req!r}")

    atoms, fixed = _load_structure(raw["structure"], base)
    pseudo_dir, pseudo_map = _resolve_pseudopotentials(
        raw["pseudopotentials"], base, atoms.get_chemical_symbols())

    kp = raw.get("kpoints", {})
    _check_keys("kpoints", kp, {"mesh", "shift"})
    sm = raw.get("smearing", {})
    _check_keys("smearing", sm, {"type", "width"})
    scf_raw = dict(raw.get("scf", {}))
    _check_keys("scf", scf_raw,
                {"max_iter", "etol", "rhotol", "mixing", "diago", "trace",
                 "convergence", "entol", "eigensolver", "magnetic"})
    mix_raw = dict(scf_raw.pop("mixing", {}))
    mag_raw = dict(scf_raw.pop("magnetic", {}))
    _check_keys("scf.magnetic", mag_raw,
                {"mixer", "spin_precond", "mixing_alpha", "diago_schedule"})
    diago = scf_raw.pop("diago", {})
    _check_keys("scf.diago", diago, {"tol"})

    xc, hybrid = _resolve_xc(raw)
    task = raw.get("task", "scf")
    if task not in ("scf", "relax", "bands", "magnetism", "eos", "elastic",
                    "phonons"):
        raise InputError(
            f"unknown task {task!r} "
            f"(scf | relax | bands | magnetism | eos | elastic | phonons)")
    nspin = int(raw.get("nspin", 1))
    if nspin not in (1, 2):
        raise InputError(f"nspin must be 1 or 2, got {nspin}")

    noncollinear, nonmagnetic, symmetry = _resolve_symmetry(raw, task)

    # selective dynamics lowers the crystal symmetry: held atoms and their free
    # symmetry partners no longer transform into each other, so IBZ reduction and
    # force symmetrization would couple a fixed atom's force back onto a partner
    # that is meant to relax. Default symmetry off when atoms are fixed and reject
    # an explicit symmetry: true here, where the message names the fix (mirrors
    # the noncollinear handling in _resolve_symmetry).
    if fixed is not None:
        if raw.get("symmetry") is True:
            raise InputError(
                "symmetry: true is invalid with selective dynamics "
                "(structure.fixed) — fixed atoms lower the crystal symmetry, so "
                "IBZ reduction and force symmetrization would couple a held atom "
                "to its free symmetry partners. Set symmetry: false (the default "
                "when atoms are fixed).")
        symmetry = False

    # a hybrid SCF is norm-conserving and spin-unpolarized (the Fock hook builds
    # on PBE exchange, nspin=1); reject the combinations the driver cannot run
    # here, where the message points at the fix rather than deep in the SCF.
    if hybrid.enabled and (nspin != 1 or noncollinear):
        raise InputError(
            f"hybrid functionals (xc: {hybrid.name}) are spin-unpolarized "
            f"(nspin=1, collinear) only")

    # distributed (k-point-sharded) SCF is a torchrun launch; the authoritative
    # gates live in api.run_scf / api.run, but they only fire at run time inside
    # torchrun. Mirror them here so `gradwave validate` rejects an unsupported
    # combination up front, with a message that names the fix. USPP/PAW is a
    # SUPPORTED formalism (this layer cannot tell it from norm-conserving anyway),
    # so nothing keys on the pseudopotential kind.
    distributed = bool(raw.get("distributed", False))
    if distributed:
        if noncollinear:
            raise InputError(
                "distributed: true is not supported with a noncollinear/SOC SCF "
                "— the k-point-sharded SCF is wired for the norm-conserving and "
                "USPP/PAW collinear paths only (see docs/manual/distributed.md)")
        if hybrid.enabled:
            raise InputError(
                f"distributed: true is not supported with a hybrid functional "
                f"(xc: {hybrid.name}) — the k-point-sharded SCF is wired for the "
                f"collinear semilocal paths only (see docs/manual/distributed.md)")
        if task not in ("scf", "bands", "relax", "eos"):
            raise InputError(
                f"distributed: true is only wired for task: scf | bands | "
                f"relax | eos (got task: {task!r}) — elastic/phonons/magnetism "
                f"don't route through the k-point-sharded SCF path yet (see "
                f"docs/manual/distributed.md)")

    # fixed spin moment: an integer-occupation pin, so only a collinear nspin=2
    # run without smearing consumes it (the calculator/SCF requirement).
    tot_mag = raw.get("tot_magnetization")
    if tot_mag is not None:
        tot_mag = float(tot_mag)
        if nspin != 2:
            raise InputError(
                "tot_magnetization (fixed spin moment M=N↑−N↓) applies only to "
                "a collinear spin run (nspin: 2)")

    # DFT+U: a per-species Dudarev correction inside the SCF. The collinear
    # (nspin 1/2) and noncollinear/spin-orbit norm-conserving spinor paths are
    # both wired (the latter via the 2×2 spin-block occupation matrix,
    # core.hubbard.occupation_matrices_noncollinear) — note this input layer
    # cannot distinguish NC from USPP/PAW pseudopotentials (that requires
    # parsing the UPF files, done later in api.build_system), but `api.run_scf`
    # only ever routes `noncollinear: true` to the norm-conserving driver
    # regardless of pseudopotential kind, so no combination reachable through
    # this Input/api.run path is left silently wrong; the USPP/PAW-noncollinear
    # SCF (reached only by calling scf.uspp_noncollinear.scf_uspp_noncollinear
    # directly, not through Input) rejects a hubbard argument explicitly there.
    # A hybrid's Fock SCF has no +U hook, so that combination is still rejected
    # at load.
    hubbard = _build_hubbard(raw.get("hubbard"), atoms.get_chemical_symbols())
    if hubbard.enabled and hybrid.enabled:
        raise InputError(
            "DFT+U (hubbard) and a hybrid functional (xc: pbe0/hse) cannot be "
            "combined; the hybrid Fock SCF has no +U hook")

    mesh = tuple(kp.get("mesh", (1, 1, 1)))
    if len(mesh) != 3:
        raise InputError(f"kpoints.mesh must have 3 entries, got {list(mesh)}")
    smtype = sm.get("type", "none")
    if smtype not in ("none", "fermi-dirac", "gaussian", "mp1", "cold"):
        raise InputError(f"unknown smearing type {smtype!r}")

    _validate_mixing(mix_raw)

    out_raw = raw.get("output", {})
    _check_keys("output", out_raw,
                {"dir", "checkpoint", "wavefunctions", "error_estimate", "volumetric"})
    volumetric = _build_volumetric(out_raw.get("volumetric", False))
    restart = raw.get("restart")

    nbands = raw.get("nbands", "auto")
    ecut = float(raw["ecut"])
    if ecut <= 0.0:
        raise InputError(f"ecut must be > 0 eV, got {ecut}")
    ecutrho = raw.get("ecutrho")
    if ecutrho is not None:
        ecutrho = float(ecutrho)
        if ecutrho <= ecut:
            raise InputError(
                f"ecutrho must exceed ecut (the density cutoff is a finer grid "
                f"than the wavefunction cutoff), got ecutrho={ecutrho} eV, "
                f"ecut={ecut} eV")
    projections = _build_projections(raw.get("projections", False))
    dispersion = _build_dispersion(raw.get("dispersion", False))
    return Input(
        atoms=atoms,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        ecut=ecut,
        ecutrho=ecutrho,
        xc=xc,
        hybrid=hybrid,
        hubbard=hubbard,
        kpoints=KPointsParams(
            mesh=mesh, shift=tuple(kp.get("shift", (0, 0, 0)))
        ),
        smearing=SmearingParams(type=smtype, width=float(sm.get("width", 0.1))),
        nbands=None if nbands == "auto" else int(nbands),
        symmetry=symmetry,
        nspin=nspin,
        noncollinear=noncollinear,
        nonmagnetic=nonmagnetic,
        start_mag=raw.get("start_mag"),
        tot_magnetization=tot_mag,
        scf=SCFParams(
            max_iter=int(scf_raw.get("max_iter", 100)),
            etol=float(scf_raw.get("etol", 1e-7)),
            rhotol=float(scf_raw.get("rhotol", 1e-7)),
            mixing=MixingParams(**mix_raw) if mix_raw else MixingParams(),
            diago_tol=float(diago.get("tol", 1e-9)),
            trace=bool(scf_raw.get("trace", False)),
            convergence=str(scf_raw.get("convergence", "density")),
            entol=float(scf_raw.get("entol", 1e-6)),
            eigensolver=str(scf_raw.get("eigensolver", "davidson")),
            magnetic=MagneticParams(**mag_raw) if mag_raw else MagneticParams(),
        ),
        task=task,
        relax=_build(RelaxParams, raw.get("relax", {}), "relax"),
        bands=_build(BandsParams, raw.get("bands", {}), "bands"),
        magnetism=_build(MagnetismParams, raw.get("magnetism", {}), "magnetism"),
        eos=_build(EOSParams, raw.get("eos", {}), "eos"),
        elastic=_build(ElasticParams, raw.get("elastic", {}), "elastic"),
        phonons=_build(PhononParams, raw.get("phonons", {}), "phonons"),
        projections=projections,
        dispersion=dispersion,
        device=raw.get("device", "cpu"),
        distributed=distributed,
        verbose=bool(raw.get("verbose", True)),
        output_dir=base / out_raw.get("dir", "./out"),
        output_checkpoint=bool(out_raw.get("checkpoint", True)),
        output_wavefunctions=bool(out_raw.get("wavefunctions", False)),
        output_volumetric=volumetric,
        error_estimate=bool(out_raw.get("error_estimate",
                                        raw.get("error_estimate", True))),
        restart=None if restart is None else (base / restart),
        fixed=fixed,
    )
