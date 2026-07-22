"""Volumetric export — the CHGCAR / PARCHG / ELF analogs for VESTA & Ovito.

After an SCF gradwave holds ρ(r) and the plane-wave coefficients c_nk(G) in
memory; this module turns them into the standard volumetric formats (.cube,
.xsf) that crystallography viewers read. The file encoding — units, voxel
order, the periodic wrap-around plane — is delegated to ASE (a core
dependency), so this module only does the physics: reading ρ(r) off a result,
reconstructing |ψ_nk(r)|² from the stored coefficients, assembling ELF(r) from
τ, ρ and |∇ρ|², and (for noncollinear runs) the magnetization density m(r).

Conventions (see core/fftbox.py): ψ_nk(r) = Ω^{-1/2} Σ_G c(G) e^{i(k+G)·r} with
Σ_G |c(G)|² = 1, so g_to_r(c) = Σ_G c e^{iGr} and the normalized single-state
density is |ψ_nk(r)|² = |g_to_r(c)|² / Ω, which integrates to 1 over the cell.
Summing w_k f_nk |ψ_nk(r)|² over occupied states reproduces res.rho.

Result coverage (see the manual, "Volumetric export"):
  * collinear norm-conserving  — density, band_density, elf all supported.
  * noncollinear / SOC         — density (total), band_density (spinor, the
                                 up/down components summed), and magnetization;
                                 elf is not yet implemented for spinors.
  * USPP / PAW                 — density (the full augmented ρ); band_density is
                                 the SOFT pseudo-density (no augmentation, as in
                                 VASP's PARCHG); elf is not yet supported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import g_to_r
from gradwave.core.metagga import tau_b
from gradwave.postscf._response import pad_coeffs

_WRITERS = {".cube": "cube", ".xsf": "xsf"}


def _is_spinor(res) -> bool:
    """NC/SOC results store spinor coeffs as one (nk,nb,2·npw_max) tensor;
    collinear results store a per-k list."""
    return res.coeffs is not None and not isinstance(res.coeffs, list)


def _atoms_from_system(system):
    """ASE Atoms (cell rows a_i [Å], Cartesian positions, true Z) from a System."""
    from ase import Atoms
    from ase.data import atomic_numbers

    cell = np.asarray(system.grid.cell, dtype=float)
    pos = system.positions.detach().cpu().numpy()
    # norm-conserving systems carry `upfs`; USPP/PAW systems carry `paws`.
    pseudos = getattr(system, "upfs", None)
    if pseudos is None:
        pseudos = system.paws
    numbers = [atomic_numbers[pseudos[s].element] for s in system.species_of_atom]
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)


def _infer_fmt(path) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _WRITERS:
        raise ValueError(
            f"unknown volumetric extension {ext!r}; use one of {sorted(_WRITERS)}"
        )
    return _WRITERS[ext]


def write_volumetric(path, data, atoms, fmt: str | None = None) -> str:
    """Write a scalar field `data` (n1,n2,n3) on `atoms`' cell to .cube/.xsf.

    The format is taken from the extension unless `fmt` ("cube"/"xsf") is given.
    Returns the path written. ASE fixes the units (Bohr for .cube, Å for .xsf)
    and the voxel ordering; the array must be indexed [i,j,k] along the cell
    rows a₁,a₂,a₃, which is exactly how gradwave stores its grids.
    """
    fmt = fmt or _infer_fmt(path)
    arr = np.ascontiguousarray(np.asarray(data, dtype=float))
    if arr.ndim != 3:
        raise ValueError(f"volumetric data must be 3-D, got shape {arr.shape}")
    with open(path, "w") as fh:
        if fmt == "cube":
            from ase.io.cube import write_cube

            write_cube(fh, atoms, data=arr)
        elif fmt == "xsf":
            from ase.io.xsf import write_xsf

            write_xsf(fh, [atoms], data=arr)
        else:
            raise ValueError(f"unknown format {fmt!r}; use 'cube' or 'xsf'")
    return str(path)


def _grid_info(res):
    system = res.system
    return system, system.grid.shape, float(system.grid.volume)


# --- CHGCAR analog: total / spin density ----------------------------------

def density(res, spin: int | None = None) -> np.ndarray:
    """ρ(r) [e/Å³] as a numpy array (the CHGCAR analog).

    spin=0/1 selects the ↑/↓ channel of a collinear spin-polarized result; the
    default (None) returns the total density, which is defined for every result
    type. Noncollinear runs have no ↑/↓ channels — use magnetization() for m(r).
    """
    if spin is None:
        rho = res.rho
    else:
        rho_spin = getattr(res, "rho_spin", None)
        if rho_spin is None:
            if getattr(res, "m", None) is not None:
                raise ValueError(
                    "noncollinear result has no ↑/↓ channels; use magnetization() for m(r)"
                )
            raise ValueError("spin channel requested but the result is not spin-polarized")
        rho = rho_spin[spin]
    return rho.detach().cpu().numpy()


def write_density(res, path, spin: int | None = None, fmt: str | None = None) -> str:
    """Write the SCF density ρ(r) to a .cube/.xsf file (CHGCAR analog)."""
    return write_volumetric(path, density(res, spin), _atoms_from_system(res.system), fmt)


# --- PARCHG analog: band/k-decomposed density -----------------------------

def band_density(res, band: int, kpoint: int = 0, spin: int | None = None) -> np.ndarray:
    """|ψ_{n,k}(r)|² [Å⁻³] for one band and k-point (the PARCHG analog).

    The single-state density integrates to 1 over the cell. `band` and
    `kpoint` index res.eigenvalues. For a collinear spin-polarized result
    `spin` picks the ↑/↓ channel; for a noncollinear/SOC result the two spinor
    components are summed, |ψ|² = |ψ↑|² + |ψ↓|², and `spin` is ignored.

    For USPP/PAW this is the SOFT pseudo-density (the augmentation charge is
    not added back), matching VASP's PARCHG.
    """
    system, shape, vol = _grid_info(res)
    if res.coeffs is None:
        raise ValueError(
            "result carries no wavefunction coefficients (coeffs is None); "
            "band_density needs the retained ψ_nk(G)"
        )
    idx = system.spheres[kpoint].flat_idx
    npw_k = idx.shape[0]
    if _is_spinor(res):
        m_pw = res.coeffs.shape[-1] // 2  # up = [:m_pw], down = [m_pw:]
        row = res.coeffs[kpoint, band]
        pu = g_to_r(row[:npw_k], idx, shape)
        pd = g_to_r(row[m_pw : m_pw + npw_k], idx, shape)
        dens = pu.real**2 + pu.imag**2 + pd.real**2 + pd.imag**2
    else:
        coeffs = res.coeffs
        if getattr(res, "nspin", 1) == 2:
            ck = coeffs[0 if spin is None else spin][kpoint][band]
        else:
            ck = coeffs[kpoint][band]
        psi = g_to_r(ck[:npw_k], idx, shape)
        dens = psi.real**2 + psi.imag**2
    return (dens / vol).detach().cpu().numpy()


def write_band_density(
    res, path, band: int, kpoint: int = 0, spin: int | None = None, fmt: str | None = None
) -> str:
    """Write |ψ_{n,k}(r)|² for a chosen band/k to .cube/.xsf (PARCHG analog)."""
    return write_volumetric(
        path, band_density(res, band, kpoint, spin), _atoms_from_system(res.system), fmt
    )


# --- Magnetization density (noncollinear) ---------------------------------

def magnetization(res, component: str = "abs") -> np.ndarray:
    """Spin magnetization density m(r) [μ_B/Å³] — noncollinear/SOC results only.

    component: 'x'/'y'/'z' for a Cartesian channel, 'abs' for the magnitude
    |m(r)|. The Cartesian components integrate to res.mag_vec. Collinear runs
    expose the scalar spin density through density(spin=0)−density(spin=1).
    """
    m = getattr(res, "m", None)
    if m is None:
        raise ValueError(
            "magnetization density is only available for noncollinear results (res.m); "
            "for a collinear spin-polarized run take density(spin=0) − density(spin=1)"
        )
    axes = {"x": 0, "y": 1, "z": 2}
    if component == "abs":
        field = (m[0] ** 2 + m[1] ** 2 + m[2] ** 2).sqrt()
    elif component in axes:
        field = m[axes[component]]
    else:
        raise ValueError(f"component must be 'x', 'y', 'z' or 'abs', got {component!r}")
    return field.detach().cpu().numpy()


def write_magnetization(
    res, path, component: str = "abs", fmt: str | None = None
) -> str:
    """Write the noncollinear magnetization density m(r) to .cube/.xsf."""
    return write_volumetric(
        path, magnetization(res, component), _atoms_from_system(res.system), fmt
    )


# --- ELF: electron localization function ----------------------------------

_C_F = 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0)  # Thomas–Fermi kinetic constant


def elf(res, eps: float = 1e-10) -> np.ndarray:
    """Becke–Edgecombe electron localization function ELF(r) ∈ [0,1].

    ELF = 1 / (1 + (D/D_h)²) with the Pauli kinetic-energy density
    D = τ − |∇ρ|²/(8ρ) and its uniform-electron-gas reference D_h = c_F ρ^{5/3}.
    ELF → 1 marks strong localization (covalent bonds, lone pairs); ELF ≈ ½ is
    the homogeneous-gas value. τ(r) comes straight from the coefficients (the
    existing meta-GGA tau_b), so no meta-GGA run is needed to produce it.

    Collinear norm-conserving, spin-unpolarized results only for now.
    """
    system, shape, vol = _grid_info(res)
    if getattr(res, "occupations", None) is None or getattr(system, "batch", None) is None:
        raise NotImplementedError(
            "ELF is implemented for collinear norm-conserving results; "
            "noncollinear and USPP/PAW support lands later"
        )
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("ELF for nspin=2 lands next — spin-unpolarized only for now")
    coeffs = pad_coeffs(res.coeffs, system.batch.npw_max)
    tau = tau_b(coeffs, res.occupations, system.kweights, system.batch, shape, vol)
    rho = res.rho
    sigma = sigma_from_rho(rho, system.grid.g_cart)  # |∇ρ|²(r)

    rho_c = rho.clamp_min(eps)
    d = tau - sigma / (8.0 * rho_c)
    d_h = _C_F * rho_c ** (5.0 / 3.0)
    chi = d / (d_h + eps)
    return (1.0 / (1.0 + chi * chi)).detach().cpu().numpy()


def write_elf(res, path, fmt: str | None = None) -> str:
    """Write the electron localization function ELF(r) to .cube/.xsf."""
    return write_volumetric(path, elf(res), _atoms_from_system(res.system), fmt)
