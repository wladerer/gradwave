"""run_scf and its noncollinear/hybrid variants."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from gradwave.api._common import (
    _DEFAULT_MIXING_HISTORY,
    SPIN_XC_REGISTRY,
    XC_REGISTRY,
    _mixing_scheme,
)
from gradwave.api.system import (
    _hubbard_manifolds,
    _is_uspp,
    _species_upfs,
    _spin_setup,
    build_system,
)
from gradwave.core.xc.base import XCFunctional
from gradwave.inputs import Input, InputError
from gradwave.scf.options import (
    BoundaryOptions,
    HubbardOptions,
    MemoryOptions,
    MixerOptions,
    SCFOptions,
    SpinOptions,
)

if TYPE_CHECKING:

    from gradwave.distributed import DistKContext
    from gradwave.scf.loop import SCFResult, System
    from gradwave.scf.noncollinear import NCResult
    from gradwave.scf.results import USPPResult
    from gradwave.scf.uspp_setup import USPPSystem

logger = logging.getLogger(__name__)


def run_scf(
    inp: Input,
    system: System | USPPSystem | None = None,
    verbose: bool = True,
    start_from: Any = None,
) -> SCFResult | NCResult | USPPResult:
    """Run the SCF for either formalism. Returns the native result
    (SCFResult for NC, USPPResult for USPP/PAW).

    ``start_from`` warm-starts the density from a previous converged result
    (the volume-scan chain in ``run_eos`` uses it); when None the checkpoint in
    ``inp.restart`` is used instead, if any.

    The ``scf.memory`` knobs travel as ARGUMENTS (Input → MemoryOptions →
    scf()/scf_uspp()/scf_noncollinear() → solvers.davidson / core.batch); the
    ``GRADWAVE_*`` env vars remain the documented user-facing overrides,
    layered over the passed values at solve time. The former api → environment
    bridge (``_davidson_memory_env``) is retired — config via os.environ
    mutation was a measured incident class (an import-frozen read silently
    ignoring it poisoned a benchmark ladder)."""
    return _run_scf(inp, system, verbose, start_from)


def _memory_options(inp: Input, *, k_levers: bool = True) -> MemoryOptions:
    """The ``scf.memory`` block as a MemoryOptions for the SCF drivers.

    ``k_levers=False`` strips ``k_chunk``/``k_parallel`` for the branches that
    never threaded them (hybrid Fock couples orbitals across k; the spinor
    driver has no k-streamed solve) — those Input fields were always silently
    inert there, and stripping keeps that exact behaviour rather than turning
    an old no-op into a new error. The default "complex128" storage maps to
    None so a default Input hands the drivers an all-None group (the
    byte-for-byte historical solve)."""
    mem = inp.scf.memory
    storage = mem.subspace_storage if mem.subspace_storage != "complex128" else None
    return MemoryOptions(
        k_chunk=mem.k_chunk if k_levers else None,
        k_parallel=mem.k_parallel if k_levers else None,
        max_dim_factor=mem.max_dim_factor,
        subspace_budget_gb=mem.subspace_budget_gb,
        subspace_storage=storage,
        dense_budget_gb=mem.dense_budget_gb,
    )


def _run_scf(
    inp: Input,
    system: System | USPPSystem | None = None,
    verbose: bool = True,
    start_from: Any = None,
) -> SCFResult | NCResult | USPPResult:
    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    system = system or build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    dist_ctx: DistKContext | None = None
    if inp.distributed:
        if inp.hybrid.enabled or inp.noncollinear:
            raise NotImplementedError(
                "distributed (k-point-sharded) SCF is implemented for the "
                "norm-conserving and USPP/PAW collinear paths (DFT+U "
                "included) — hybrid functionals and the noncollinear/SOC SCF "
                "are documented follow-ups (see docs/manual/distributed.md)"
            )
        from gradwave.distributed import init_from_env, shard_system, shard_uspp_system

        info = init_from_env()
        if info is None and type(inp.distributed) is not bool:
            # int rank count reached the api WITHOUT a torchrun env: only the
            # CLI self-launches (cli._maybe_relaunch_ranks). Refuse rather
            # than silently running serial — the user asked for N ranks.
            raise RuntimeError(
                f"distributed: {inp.distributed} (local ranks) requires the "
                "CLI (`gradwave input.yaml`, which self-launches torchrun) "
                "or an explicit torchrun launch — a plain api.run()/run_scf() "
                "call has no rank environment to shard over")
        if info is not None:
            rank, world_size, group = info
            # `distributed: N` (int local-rank count) under an EXISTING
            # torchrun env must agree with the launched world size — a
            # mismatch means the user asked for one layout and launched
            # another (the CLI self-launch always matches; a manual torchrun
            # with a stale YAML is the case this catches).
            if type(inp.distributed) is not bool and int(inp.distributed) != world_size:
                raise ValueError(
                    f"distributed: {inp.distributed} (local ranks) does not "
                    f"match the launched torchrun world size {world_size} — "
                    "either fix the YAML to the launched rank count or use "
                    "`distributed: true` to accept the env's layout")
            # symmetry (IBZ) composes with the sharding: shard_* slices the
            # IBZ k-list and the symmetrizers ride along — nothing to gate.
            if uspp:
                system, dist_ctx = shard_uspp_system(
                    cast("USPPSystem", system), rank, world_size, group)
            else:
                system, dist_ctx = shard_system(
                    cast("System", system), rank, world_size, group)
            # every rank still computes the identical, correctly-reduced
            # result (see scf.loop.scf / scf.uspp_loop.scf_uspp dist_ctx
            # handling) — quiet every rank but 0 purely to avoid world_size-fold
            # duplicated chatter
            verbose = verbose and rank == 0
    if inp.hybrid.enabled:
        return _run_scf_hybrid(inp, system, verbose, start_from, uspp)
    if inp.noncollinear:
        return _run_scf_noncollinear(inp, system, verbose)
    if inp.nspin == 2:
        xc, mags = _spin_setup(inp)
    else:
        xc = XC_REGISTRY[inp.xc]()
        mags = None

    if start_from is None and inp.restart is not None:
        from gradwave.io.checkpoint import as_start_from, load_checkpoint

        start_from = as_start_from(load_checkpoint(inp.restart))

    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    # DFT+U: the same manifold list feeds the NC and USPP/PAW SCF (both take a
    # `hubbard=` kwarg — a runtime input, so it stays a flat argument); the
    # convergence aids ride in the HubbardOptions group. Defaults (β=1.0, ramp
    # off) leave today's numbers bit-for-bit.
    manifolds = _hubbard_manifolds(inp)
    hub_grp = None
    if manifolds is not None:
        hub_grp = HubbardOptions(occ_mix=inp.hubbard.occ_mix,
                                 u_ramp_iters=inp.hubbard.u_ramp_iters)
    if inp.tot_magnetization is not None and uspp:
        # fixed spin moment: a collinear nspin=2 pin. scf_uspp has no such
        # mode; error rather than silently running an unconstrained SCF that
        # would masquerade as a fixed-moment result.
        raise NotImplementedError(
            "tot_magnetization (fixed spin moment) is norm-conserving only "
            "— the USPP/PAW SCF driver has no fixed-moment mode yet")
    # The shared scalar config both drivers consume identically. dict[str, Any]
    # for the same reason the old flat-kwarg literal used it: the value union
    # would otherwise be checked key-blind against SCFOptions' fields.
    common: dict[str, Any] = dict(
        smearing=inp.smearing.type, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        diago_tol=inp.scf.diago_tol, verbose=verbose,
        # energy-metric convergence gate (opt-in via scf.convergence:
        # "energy"); "density" leaves the density gate bit-for-bit unchanged.
        energy_metric=(inp.scf.convergence == "energy"), entol=inp.scf.entol,
        eigensolver=inp.scf.eigensolver,
        hubbard=hub_grp,
    )
    # uspp (from _is_uspp(upfs) above) already tells us which concrete type
    # `system` is — the two branches below just name that for the checker.
    if uspp:
        from gradwave.scf.uspp import scf_uspp

        # scf_uspp has no eigensolver knob — the S-metric problem is Davidson-only.
        # Reject an explicit chebyshev request here, mirroring the calculator's
        # rejection (calculator.calculate). "auto" is fine: it resolves to davidson
        # for the USPP path (CheFSI only gates in on the NC standard problem).
        if inp.scf.eigensolver == "chebyshev":
            raise InputError(
                "scf.eigensolver='chebyshev' is norm-conserving only; the "
                "USPP/PAW generalized S-metric problem is not supported yet")
        opts = SCFOptions(
            **common,
            # history=None keeps the per-scheme default (johnson 12, else 8);
            # scheme auto → None so the formalism's resolver picks its
            # evidence-backed default (_resolve_uspp_mixing_scheme)
            mixer=MixerOptions(alpha=inp.scf.mixing.alpha,
                               history=inp.scf.mixing.history,
                               scheme=_mixing_scheme(inp), kerker=kerker,
                               precond=inp.scf.mixing.precond),
            spin=SpinOptions(nspin=inp.nspin, start_mag=mags),
            # target_mu is NC-only and was never threaded on this branch — an
            # Input setting it on a USPP run stays the historical silent no-op
            # here (scf_uspp itself rejects a non-None value loudly).
            boundary=BoundaryOptions(kind=inp.scf.boundary,
                                     esm_bias=inp.scf.esm_bias),
            # only dense_budget_gb: the subspace/k knobs are NC-only and the
            # USPP generalized Davidson never read them (the old env bridge
            # exported them unread) — scf_uspp rejects non-None values, so
            # strip rather than convert a historical no-op into an error.
            memory=MemoryOptions(dense_budget_gb=inp.scf.memory.dense_budget_gb),
        )
        return scf_uspp(cast("USPPSystem", system), xc, opts=opts,
                        hubbard=manifolds, start_from=start_from,
                        dist_ctx=dist_ctx)
    from gradwave.scf.loop import scf

    opts = SCFOptions(
        **common,
        mixer=MixerOptions(
            alpha=inp.scf.mixing.alpha,
            # the NC mixers take a fixed integer history (None → the shared
            # default the api has always supplied)
            history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
            scheme=_mixing_scheme(inp), kerker=kerker,
            precond=inp.scf.mixing.precond),
        spin=SpinOptions(nspin=inp.nspin, start_mag=mags,
                         tot_magnetization=inp.tot_magnetization),
        boundary=BoundaryOptions(kind=inp.scf.boundary,
                                 esm_bias=inp.scf.esm_bias,
                                 target_mu=inp.scf.target_mu),
        # k-streaming/k-parallel + Davidson subspace + dense-box budget; None
        # fields keep the env-default fallbacks (_resolve_k_chunk /
        # _resolve_k_parallel / solvers.davidson's per-solve resolvers).
        memory=_memory_options(inp),
    )
    return scf(cast("System", system), cast("XCFunctional", xc), opts=opts,
               hubbard=manifolds, start_from=start_from, dist_ctx=dist_ctx)


def _run_scf_noncollinear(
    inp: Input, system: System | USPPSystem, verbose: bool
) -> NCResult:
    """A plain non-collinear (spinor) SCF for task: scf with
    noncollinear: true. Builds a NoncollinearXC from inp.xc (as run_magnetism
    does), seeds the atomic moments along +z from start_mag (or warm-starts
    from a checkpoint's m⃗ field), and returns the NCResult."""
    import torch

    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.scf.noncollinear import scf_noncollinear

    xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())

    if inp.nonmagnetic:
        # spin-orbit only: pin m⃗ ≡ 0 (no seed, no spurious moment)
        mag_vec_init = torch.zeros((len(inp.atoms), 3), dtype=torch.float64)
    elif inp.restart is not None:
        from gradwave.io.checkpoint import load_checkpoint, nc_mag_seed

        mag_vec_init = nc_mag_seed(load_checkpoint(inp.restart), system)
    else:
        # high-spin seed along +z; per-species magnitude from start_mag (a
        # moment fraction ~ scale on the SAD magnetization), default 1.0
        symbols = inp.atoms.get_chemical_symbols()
        z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        scales = [float((inp.start_mag or {}).get(s, 1.0)) for s in symbols]
        mag_vec_init = torch.stack([s * z for s in scales])  # (na, 3)

    # NC SCF requires a real smearing scheme (spinor bands hold one electron)
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    # DFT+U: noncollinear/spin-orbit +U (the 2×2 spin-block occupation matrix,
    # core.hubbard) — same manifold list the collinear/USPP paths take.
    manifolds = _hubbard_manifolds(inp)
    # run_scf routes noncollinear: true to this (norm-conserving) driver
    # regardless of pseudopotential kind (see inputs.py's HubbardParams
    # docstring) — scf_noncollinear's own System-only annotation is the
    # narrow spot, not this call.
    return scf_noncollinear(
        cast("System", system), xc, mag_vec_init=mag_vec_init,
        smearing=smtype, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
        # magnetic-channel spinor knobs (scf.magnetic); the spinor driver
        # resolves its (ρ, m⃗) mixing independently of scf.mixing.scheme, so
        # these are the only path to them from an input file.
        mag_mixer=inp.scf.magnetic.mixer,
        spin_precond=inp.scf.magnetic.spin_precond,
        mag_mixing_alpha=inp.scf.magnetic.mixing_alpha,
        mag_diago_schedule=inp.scf.magnetic.diago_schedule,
        # energy-metric convergence gate (opt-in via scf.convergence: "energy");
        # the default "density" leaves the residual gate bit-for-bit unchanged.
        energy_metric=(inp.scf.convergence == "energy"), entol=inp.scf.entol,
        diago_tol=inp.scf.diago_tol, verbose=verbose,
        nonmagnetic=inp.nonmagnetic,
        hubbard=manifolds,
        # scf.memory subspace/dense knobs → spinor eigensolve arguments (the
        # old env bridge applied them here too); k levers stripped — the
        # spinor driver has no k-streamed solve and those Input fields were
        # always inert on this branch.
        memory=_memory_options(inp, k_levers=False),
    )


def _run_scf_hybrid(
    inp: Input,
    system: System | USPPSystem,
    verbose: bool,
    start_from: Any,
    uspp: bool,
) -> SCFResult:
    """Self-consistent PBE0-form / screened hybrid SCF (xc: pbe0 | hse).

    Dispatches to ``postscf.hybrid.hybrid_scf``, which scales the semilocal PBE
    exchange by (1−α) and adds α·E_x^Fock through the SCF ``fock`` hook (the
    multi-k build, full BZ). Norm-conserving, nspin=1 (the input layer already
    rejected the other combinations); the system was built full-BZ in
    ``build_system``. Returns the same SCFResult as a plain NC run."""
    if uspp:
        raise NotImplementedError(
            "hybrid functionals need norm-conserving pseudopotentials "
            "(the Fock hook builds on the norm-conserving exchange path)")
    from gradwave.postscf.hybrid import hybrid_scf

    if start_from is None and inp.restart is not None:
        from gradwave.io.checkpoint import as_start_from, load_checkpoint

        start_from = as_start_from(load_checkpoint(inp.restart))

    hy = inp.hybrid
    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    omega = hy.omega if hy.mode != "full" else None
    # the uspp guard above rules out USPPSystem; hybrid_scf is norm-conserving only.
    return hybrid_scf(
        cast("System", system), alpha=hy.alpha, mode=hy.mode, omega=omega,
        smearing=inp.smearing.type, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
        kerker=kerker, diago_tol=inp.scf.diago_tol,
        start_from=start_from, verbose=verbose,
        # scf.memory subspace/dense knobs, forwarded through hybrid_scf's
        # **scf_kwargs to scf(memory=...) (the old env bridge applied them on
        # this branch too); k levers stripped — hybrid Fock couples orbitals
        # across k, and those Input fields were always inert here.
        memory=_memory_options(inp, k_levers=False))
