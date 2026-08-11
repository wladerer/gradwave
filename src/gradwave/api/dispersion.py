"""D3/D4 dispersion terms folded into summaries and phonon forces."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from gradwave.api._common import _get
from gradwave.inputs import Input

if TYPE_CHECKING:


    from gradwave.api._common import SCFLike

logger = logging.getLogger(__name__)


class _DispersionTerms(NamedTuple):
    """Evaluated D3/D4 correction: energy (eV scalar), forces (na,3), stress
    (3,3) or None when not requested, and the resolved config (carries the
    damping params s6/s8/a1/a2). Returned by :func:`_compute_dispersion`."""

    energy: Any
    forces: Any
    stress: Any
    cfg: Any


def _compute_dispersion(
    positions: Any, cell: Any, z: list[int], *,
    method: str, functional: str, charge: float = 0.0,
    cutoff_ang: float, cn_cutoff_ang: float,
    s6: float | None = None, s8: float | None = None,
    a1: float | None = None, a2: float | None = None,
    need_stress: bool = True,
) -> _DispersionTerms:
    """Resolve the D3(BJ)/D4(BJ) config and evaluate energy, forces, and (if
    ``need_stress``) stress. The single shared compute core behind both
    ``_apply_dispersion`` (YAML pipeline, summary-dict output) and
    ``GradWave._apply_dispersion`` (calculator, ASE-results output): the two
    callers differ only in how they read the knobs and shape the result, so the
    resolve+evaluate step — and any future damping-param or model change — lives
    here once. ``method`` is ``'d3'`` (default) or ``'d4'`` (``'d4'`` also uses
    ``charge``). Raises ``ValueError``/``NotImplementedError`` when the element
    set is uncovered or no BJ preset exists; callers catch and degrade.

    Each branch calls its OWN energy/forces/stress trio right after building the
    matching Config type (rather than joining afterward, which would leave a
    same-named-but-different-signature callable unioned with an incompatible
    ``D3Config | D4Config`` argument type)."""
    import torch

    cell_t = torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
    if method == "d4":
        from gradwave.postscf.dispersion_d4 import (
            D4Config,
            dispersion_energy,
            dispersion_forces,
            dispersion_stress,
        )
        cfg = D4Config.resolve(
            functional, charge=charge, cutoff_ang=cutoff_ang,
            cn_cutoff_ang=cn_cutoff_ang, s6=s6, s8=s8, a1=a1, a2=a2,
        )
        e = dispersion_energy(positions, cell_t, z, cfg)
        forces = dispersion_forces(positions, cell, z, cfg)
        stress = dispersion_stress(positions, cell, z, cfg) if need_stress else None
    else:
        from gradwave.postscf.dispersion import (
            D3Config,
            dispersion_energy,
            dispersion_forces,
            dispersion_stress,
        )
        cfg = D3Config.resolve(
            functional, cutoff_ang=cutoff_ang, cn_cutoff_ang=cn_cutoff_ang,
            s6=s6, s8=s8, a1=a1, a2=a2,
        )
        e = dispersion_energy(positions, cell_t, z, cfg)
        forces = dispersion_forces(positions, cell, z, cfg)
        stress = dispersion_stress(positions, cell, z, cfg) if need_stress else None
    return _DispersionTerms(e, forces, stress, cfg)


def _apply_dispersion(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Compute the D3(BJ)/D4(BJ) correction, fold its energy into ``res.energies``
    (so the reported total/free energy include it), and return the summary block
    (energy, forces, stress, resolved damping). ``dispersion.method`` selects
    ``'d3'`` (default) or ``'d4'`` (charge-dependent EEQ C6, periodic Ewald charge
    model). Degrades to ``{'available': False}`` when the element set is uncovered
    or no BJ preset exists for the functional."""
    import numpy as np
    import torch

    dp = inp.dispersion
    method = dp.method.lower()
    system = _get(res, "system")
    positions = system.positions.detach().to(torch.float64)
    cell = np.asarray(system.grid.cell, dtype=np.float64)
    z = [int(v) for v in inp.atoms.get_atomic_numbers()]
    try:
        e, forces, stress, cfg = _compute_dispersion(
            positions, cell, z, method=method,
            functional=dp.functional or inp.xc, charge=dp.charge,
            cutoff_ang=dp.cutoff, cn_cutoff_ang=dp.cn_cutoff,
            s6=dp.s6, s8=dp.s8, a1=dp.a1, a2=dp.a2, need_stress=True,
        )
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}

    # fold the energy into the breakdown; total/free_energy pick it up
    res.energies.dispersion = e.detach().to(positions.device)
    return {
        "available": True,
        "method": f"{method}-bj",
        "functional": (dp.functional or inp.xc).lower(),
        "damping": {"s6": cfg.s6, "s8": cfg.s8, "a1": cfg.a1, "a2_bohr": cfg.a2},
        "energy_eV": float(e),
        "energy_per_atom_eV": float(e) / len(z),
        "forces_eV_ang": forces.detach().cpu().tolist(),
        "stress_eV_ang3": stress.detach().cpu().tolist(),
    }
