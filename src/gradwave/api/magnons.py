"""run_magnons: magnon band structure via linear spin-wave theory (Layer C).

A numbers-in / dispersion-out task (like ``thermochem``): it runs no SCF. The
Heisenberg model carried in ``inp.magnons`` — spin lengths, exchange bonds
(J and DM in meV), moment directions and single-ion anisotropy — is fed through
:mod:`gradwave.postscf.magnons` on the magnetic primitive cell (``inp.atoms``),
whose Brillouin zone defines the q-path. Couplings typically come from a prior
``spin_exchange`` extraction; the physics and full docstrings live in the
postscf module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gradwave.inputs import Input, InputError


def run_magnons(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Magnon dispersion (task: magnons) — no SCF.

    Builds a :class:`gradwave.postscf.magnons.HeisenbergModel` from
    ``inp.magnons`` (converting the meV couplings to eV) on the magnetic
    primitive cell ``inp.atoms``, diagonalizes the linear-spin-wave Hamiltonian
    along the ASE band path, and returns the ``magnons`` summary block:
    frequencies in meV along the q-path, plus the ferromagnetic spin-wave
    stiffness D (meV·Å²) when the ground state is a collinear ferromagnet."""
    from gradwave.postscf.magnons import (
        HeisenbergModel,
        MagnonInstabilityError,
        magnon_bands,
        spin_wave_stiffness,
    )

    mp = inp.magnons
    n_atoms = len(inp.atoms)
    if len(mp.spins) != n_atoms:
        raise InputError(
            f"magnons.spins has {len(mp.spins)} entries but the structure has "
            f"{n_atoms} atom(s) — give one spin length S per magnetic sublattice")
    if mp.moments is not None and len(mp.moments) != n_atoms:
        raise InputError(
            f"magnons.moments has {len(mp.moments)} entries but the structure "
            f"has {n_atoms} atom(s)")
    if mp.anisotropy_k_meV is not None and len(mp.anisotropy_k_meV) != n_atoms:
        raise InputError(
            f"magnons.anisotropy_k_meV has {len(mp.anisotropy_k_meV)} entries "
            f"but the structure has {n_atoms} atom(s)")

    mev = 1.0e-3  # meV -> eV
    shells = [
        {"i": b.i, "j": b.j, "rs": [b.r], "j_iso": b.j_iso * mev,
         "dm": tuple(x * mev for x in b.dm)}
        for b in mp.bonds
    ]
    moments = None if mp.moments is None else np.asarray(mp.moments, dtype=float)
    easy_axis = np.tile(np.asarray(mp.easy_axis, dtype=float), (n_atoms, 1))
    k_ev = (None if mp.anisotropy_k_meV is None
            else np.asarray(mp.anisotropy_k_meV, dtype=float) * mev)

    model = HeisenbergModel.from_shells(
        cell=np.asarray(inp.atoms.cell.array, dtype=float),
        spins=np.asarray(mp.spins, dtype=float),
        shells=shells,
        positions=np.asarray(inp.atoms.get_scaled_positions(), dtype=float),
        moments=moments,
        anisotropy_k=k_ev,
        easy_axis=easy_axis,
    )

    try:
        bs = magnon_bands(model, path=mp.path, npoints=mp.npoints)
    except MagnonInstabilityError as exc:
        raise InputError(
            f"magnons: {exc} Set magnons.moments to the correct ordered state "
            f"(e.g. alternating directions for an antiferromagnet).") from None

    freqs = np.asarray(bs.frequencies)
    block: dict[str, Any] = {
        "n_sublattices": model.n_sub,
        "n_bonds": len(model.bonds),
        "path": mp.path or "(lattice default)",
        "qpts_frac": bs.qpts_frac.tolist(),
        "x": bs.x.tolist() if bs.x is not None else None,
        "labels": bs.labels,
        "frequencies_meV": freqs.tolist(),
        "min_frequency_meV": float(freqs.min()),
        "max_frequency_meV": float(freqs.max()),
    }

    # Ferromagnetic spin-wave stiffness D (meV·Å²) from a small-q parabolic fit,
    # reported only when every moment is collinear (a single acoustic branch
    # through ω=0). Skipped for AFM/canted states, where "the stiffness" is
    # ill-defined (multiple gapless/degenerate branches).
    mom = np.asarray(model.moments, dtype=float)
    mom = mom / np.linalg.norm(mom, axis=1, keepdims=True)
    collinear = bool(np.allclose(np.abs(mom @ mom[0]), 1.0, atol=1e-8))
    gapless = float(freqs.min()) < 1e-2  # meV
    if collinear and gapless:
        try:
            d = spin_wave_stiffness(model, direction=(1, 0, 0))
            block["stiffness_meV_A2"] = d
        except MagnonInstabilityError:
            pass

    if verbose:
        msg = (f"magnons: {model.n_sub} branch(es), "
               f"ω ∈ [{freqs.min():.3f}, {freqs.max():.3f}] meV")
        if "stiffness_meV_A2" in block:
            msg += f", D = {block['stiffness_meV_A2']:.1f} meV·Å²"
        print(msg, flush=True)
    return block
