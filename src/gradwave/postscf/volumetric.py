"""Volumetric export — the CHGCAR / PARCHG / ELF analogs for VESTA & Ovito.

After an SCF gradwave holds ρ(r) and the plane-wave coefficients c_nk(G) in
memory; this module turns them into the standard volumetric formats (.cube,
.xsf, VASP CHGCAR) that crystallography viewers read. The file encoding —
units, voxel order, the periodic wrap-around plane — is delegated to ASE (a
core dependency), so this module only does the physics: reading ρ(r) off a
result,
reconstructing |ψ_nk(r)|² from the stored coefficients, assembling ELF(r) from
τ, ρ and |∇ρ|², and (for noncollinear runs) the magnetization density m(r).

Conventions (see core/fftbox.py): ψ_nk(r) = Ω^{-1/2} Σ_G c(G) e^{i(k+G)·r} with
Σ_G |c(G)|² = 1, so g_to_r(c) = Σ_G c e^{iGr} and the normalized single-state
density is |ψ_nk(r)|² = |g_to_r(c)|² / Ω, which integrates to 1 over the cell.
Summing w_k f_nk |ψ_nk(r)|² over occupied states reproduces res.rho.

Result coverage (see the manual, "Volumetric export"):
  * collinear norm-conserving  — density, band_density, elf all supported.
  * noncollinear / SOC         — density (total), band_density (spinor, the
                                 up/down components summed), magnetization, and
                                 elf (the closed-shell ELF of the charge density).
  * USPP / PAW                 — density (the full augmented ρ); band_density is
                                 the SOFT pseudo-density (no augmentation, as in
                                 VASP's PARCHG); elf is not yet supported.
"""

from __future__ import annotations

from pathlib import Path, PosixPath
from typing import cast

import ase.atoms
import numpy as np
from torch import Tensor

from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import g_to_r
from gradwave.core.metagga import tau_b
from gradwave.postscf._response import pad_coeffs
from gradwave.scf.loop import SCFResult, System
from gradwave.scf.noncollinear import NCResult
from gradwave.scf.results import USPPNCResult, USPPResult
from gradwave.scf.uspp_setup import USPPSystem

# The four SCF drivers' result shapes; see gradwave.scf.results.AnyResult for
# the TYPE_CHECKING-only alias this mirrors. Spelled out here (rather than
# importing AnyResult) so ty's error messages name the concrete types.
AnyResult = SCFResult | NCResult | USPPResult | USPPNCResult

_WRITERS = {".cube": "cube", ".xsf": "xsf", ".chgcar": "chgcar"}


def _is_spinor(res: AnyResult) -> bool:
    """NC/SOC results store spinor coeffs as one (nk,nb,2·npw_max) tensor;
    collinear results store a per-k list."""
    return res.coeffs is not None and not isinstance(res.coeffs, list)


def _atoms_from_system(system: System | USPPSystem) -> ase.atoms.Atoms:
    """ASE Atoms (cell rows a_i [Å], Cartesian positions, true Z) from a System."""
    from ase import Atoms
    from ase.data import atomic_numbers

    cell = np.asarray(system.grid.cell, dtype=float)
    pos = system.positions.detach().cpu().numpy()
    # norm-conserving systems carry `upfs`; USPP/PAW systems carry `paws`.
    pseudos = system.paws if isinstance(system, USPPSystem) else system.upfs
    numbers = [atomic_numbers[pseudos[s].element] for s in system.species_of_atom]
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)


def _infer_fmt(path: PosixPath) -> str:
    p = Path(path)
    # CHGCAR/PARCHG conventionally carry no extension; key off the basename,
    # allowing a descriptive prefix (diamond_CHGCAR) as VASP tooling does.
    stem = p.name.upper()
    if "CHGCAR" in stem or "PARCHG" in stem:
        return "chgcar"
    ext = p.suffix.lower()
    if ext not in _WRITERS:
        raise ValueError(
            f"unknown volumetric extension {ext!r}; use one of {sorted(_WRITERS)}"
        )
    return _WRITERS[ext]


def write_volumetric(
    path: PosixPath, data: np.ndarray, atoms: ase.atoms.Atoms, fmt: str | None = None
) -> str:
    """Write a scalar field `data` (n1,n2,n3) on `atoms`' cell to .cube/.xsf.

    The format is taken from the extension unless `fmt` ("cube"/"xsf"/"chgcar")
    is given; a file named CHGCAR or PARCHG is recognized as VASP CHGCAR without
    an extension. Returns the path written. ASE fixes the units (Bohr for .cube,
    Å for .xsf) and the voxel ordering; the array must be indexed [i,j,k] along
    the cell rows a₁,a₂,a₃, which is exactly how gradwave stores its grids.

    The CHGCAR writer stores ρ·Ω following the VASP convention, so ASE's
    VaspChargeDensity reader (and tools built on it) recover ρ(r) in e/Å³.
    """
    fmt = fmt or _infer_fmt(path)
    arr = np.ascontiguousarray(np.asarray(data, dtype=float))
    if arr.ndim != 3:
        raise ValueError(f"volumetric data must be 3-D, got shape {arr.shape}")
    if fmt == "chgcar":
        from ase.calculators.vasp import VaspChargeDensity

        vcd = VaspChargeDensity(filename=None)
        vcd.atoms = [atoms]
        vcd.chg = [arr]
        vcd.write(str(path))
        return str(path)
    with open(path, "w") as fh:
        if fmt == "cube":
            from ase.io.cube import write_cube

            write_cube(fh, atoms, data=arr)
        elif fmt == "xsf":
            from ase.io.xsf import write_xsf

            write_xsf(fh, [atoms], data=arr)
        else:
            raise ValueError(f"unknown format {fmt!r}; use 'cube', 'xsf' or 'chgcar'")
    return str(path)


def _grid_info(
    res: AnyResult,
) -> tuple[System | USPPSystem, tuple[int, int, int], float]:
    system = res.system
    return system, system.grid.shape, float(system.grid.volume)


# --- CHGCAR analog: total / spin density ----------------------------------

def density(res: AnyResult, spin: int | None = None) -> np.ndarray:
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


def write_density(
    res: AnyResult, path: PosixPath, spin: int | None = None, fmt: str | None = None
) -> str:
    """Write the SCF density ρ(r) to a .cube/.xsf file (CHGCAR analog)."""
    return write_volumetric(path, density(res, spin), _atoms_from_system(res.system), fmt)


# --- PARCHG analog: band/k-decomposed density -----------------------------

def band_density(
    res: AnyResult, band: int, kpoint: int = 0, spin: int | None = None
) -> np.ndarray:
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
    coeffs_raw = res.coeffs
    if isinstance(coeffs_raw, Tensor):
        # noncollinear/SOC: one (nk, nb, 2·npw_max) spinor tensor.
        m_pw = coeffs_raw.shape[-1] // 2  # up = [:m_pw], down = [m_pw:]
        row = coeffs_raw[kpoint, band]
        pu = g_to_r(row[:npw_k], idx, shape)
        pd = g_to_r(row[m_pw : m_pw + npw_k], idx, shape)
        dens = pu.real**2 + pu.imag**2 + pd.real**2 + pd.imag**2
    else:
        # collinear: [k] per k (nspin=1) or [spin][k] (nspin=2).
        if getattr(res, "nspin", 1) == 2:
            coeffs_sp = cast("list[list[Tensor]]", coeffs_raw)
            ck = coeffs_sp[0 if spin is None else spin][kpoint][band]
        else:
            coeffs_flat = cast("list[Tensor]", coeffs_raw)
            ck = coeffs_flat[kpoint][band]
        psi = g_to_r(ck[:npw_k], idx, shape)
        dens = psi.real**2 + psi.imag**2
    return (dens / vol).detach().cpu().numpy()


def write_band_density(
    res: AnyResult, path: PosixPath, band: int, kpoint: int = 0, spin: int | None = None,
    fmt: str | None = None,
) -> str:
    """Write |ψ_{n,k}(r)|² for a chosen band/k to .cube/.xsf (PARCHG analog)."""
    return write_volumetric(
        path, band_density(res, band, kpoint, spin), _atoms_from_system(res.system), fmt
    )


# --- Magnetization density (noncollinear) ---------------------------------

def magnetization(res: AnyResult, component: str = "abs") -> np.ndarray:
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
    res: AnyResult, path: PosixPath, component: str = "abs", fmt: str | None = None
) -> str:
    """Write the noncollinear magnetization density m(r) to .cube/.xsf."""
    return write_volumetric(
        path, magnetization(res, component), _atoms_from_system(res.system), fmt
    )


# --- ELF: electron localization function ----------------------------------

_C_F = 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0)  # Thomas–Fermi kinetic constant


def _elf_field(tau: Tensor, rho: Tensor, g_cart: Tensor, c_F: float, eps: float) -> np.ndarray:
    """ELF(r) = 1/(1+(D/D_h)²) for one (τ, ρ) channel with TF constant c_F."""
    sigma = sigma_from_rho(rho, g_cart)  # |∇ρ|²(r)
    rho_c = rho.clamp_min(eps)
    d = tau - sigma / (8.0 * rho_c)
    d_h = c_F * rho_c ** (5.0 / 3.0)
    chi = d / (d_h + eps)
    return (1.0 / (1.0 + chi * chi)).detach().cpu().numpy()


def elf(res: AnyResult, eps: float = 1e-10) -> np.ndarray:
    """Becke–Edgecombe electron localization function ELF(r) ∈ [0,1].

    ELF = 1 / (1 + (D/D_h)²) with the Pauli kinetic-energy density
    D = τ − |∇ρ|²/(8ρ) and its uniform-electron-gas reference D_h = c_F ρ^{5/3}.
    ELF → 1 marks strong localization (covalent bonds, lone pairs); ELF ≈ ½ is
    the homogeneous-gas value. τ(r) comes straight from the coefficients (the
    existing meta-GGA tau_b), so no meta-GGA run is needed to produce it.

    Norm-conserving results (collinear and noncollinear). For collinear nspin=2
    the ELF is spin-resolved: each channel uses its own (τ_σ, ρ_σ) and the
    spin-scaled TF reference D_h,σ = c_F·2^{2/3}·ρ_σ^{5/3}, and the return gains a
    leading spin axis (2, n1, n2, n3). For a noncollinear/SOC spinor result the ELF
    is the closed-shell form of the CHARGE density (the trace of the spin-density
    matrix, res.rho) and the total kinetic-energy density τ_0 = τ_↑↑ + τ_↓↓ (the
    trace of the 2×2 KE-density matrix) — no spin split, so no 2^{2/3} factor;
    a single (n1, n2, n3) field, with or without spin-orbit coupling. USPP/PAW
    still raise below.
    """
    system, shape, vol = _grid_info(res)
    # USPP/PAW results carry no ``batch`` (the soft KE density would need the
    # augmentation), and a result without occupations (e.g. a bands-only NCResult)
    # cannot form τ. Everything else — collinear NC and the noncollinear spinor
    # path below — is supported.
    batch = getattr(system, "batch", None)
    occ = getattr(res, "occupations", None)
    if batch is None or occ is None:
        raise NotImplementedError(
            "ELF is implemented for norm-conserving results (collinear and "
            "noncollinear); USPP/PAW support lands later"
        )
    # Only System (not USPPSystem) carries `.batch`, so the check above
    # already excludes USPPResult/USPPNCResult (whose `.system` is always a
    # USPPSystem) -- make that narrowing explicit on `res`/`system` so the
    # rest of this function can use their norm-conserving-only fields.
    assert isinstance(res, SCFResult | NCResult) and isinstance(system, System)
    if isinstance(res, NCResult):
        # Noncollinear/SOC: spinor coeffs live on a doubled plane-wave axis. The
        # ELF uses the CHARGE density res.rho and the total (trace) KE density
        # τ_0 = τ_↑↑ + τ_↓↓ from the 2×2 KE-density matrix (spinor_tau_matrix_b),
        # i.e. the closed-shell ELF. τ_0 ≥ |∇ρ|²/(8ρ) (von Weizsäcker bound on the
        # total density) so D ≥ 0 and ELF ∈ [0, 1], as in the collinear case.
        from gradwave.core.metagga import spinor_tau_matrix_b
        assert res.coeffs is not None  # already checked (occ) not None above; coeffs pairs with it
        tau_scalar, _ = spinor_tau_matrix_b(
            res.coeffs, occ, system.kweights, batch, shape, vol, batch.npw_max)
        return _elf_field(tau_scalar, res.rho, system.grid.g_cart, _C_F, eps)
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        coeffs = pad_coeffs(cast("list[Tensor]", res.coeffs), batch.npw_max)
        tau = tau_b(coeffs, occ, system.kweights, batch, shape, vol)
        return _elf_field(tau, res.rho, system.grid.g_cart, _C_F, eps)

    # nspin=2: per-channel ELF. The uniform-gas reference for a single spin
    # channel carries the 2^{2/3} factor (each channel is its own fully spin-
    # polarized gas), the only change beyond running the loop per spin.
    assert res.rho_spin is not None  # scf() always sets rho_spin at nspin=2
    c_F_spin = _C_F * 2.0 ** (2.0 / 3.0)
    coeffs_sp = cast("list[list[Tensor]]", res.coeffs)
    fields = []
    for sp in range(nspin):
        coeffs = pad_coeffs(coeffs_sp[sp], batch.npw_max)
        tau = tau_b(coeffs, occ[sp], system.kweights, batch, shape, vol)
        fields.append(_elf_field(tau, res.rho_spin[sp], system.grid.g_cart, c_F_spin, eps))
    return np.stack(fields)


def write_elf(res: AnyResult, path: PosixPath, fmt: str | None = None) -> str | list[str]:
    """Write the electron localization function ELF(r) to .cube/.xsf.

    For nspin=2 the two spin channels are written to `<stem>_up` and `<stem>_dn`
    files and the list of paths is returned; nspin=1 writes one file."""
    field = elf(res)
    atoms = _atoms_from_system(res.system)
    if field.ndim == 3:
        return write_volumetric(path, field, atoms, fmt)
    p = Path(path)
    written = []
    for sp, tag in enumerate(("up", "dn")):
        # Path.with_name()'s stub return type is the base `Path`, not the
        # `PosixPath` write_volumetric (matching this codebase's path-like
        # convention) declares -- on this project's target platform (Linux)
        # Path(...) always constructs a real PosixPath at runtime.
        chan = cast(PosixPath, p.with_name(f"{p.stem}_{tag}{p.suffix}"))
        written.append(write_volumetric(chan, field[sp], atoms, fmt))
    return written
