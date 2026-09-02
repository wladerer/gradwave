"""Free-energy thermochemistry for gas-phase references and adsorbates.

This module turns raw DFT energies plus a list of harmonic frequencies into
temperature-dependent free energies, so an adsorption energy ``E_ads`` becomes a
free energy of adsorption ``ΔG_ads(T)`` usable in microkinetics and the
computational hydrogen electrode (CHE). It is the thermodynamic partner of
:mod:`gradwave.postscf.thermo` (whole-cell harmonic phonon-DOS thermo): where
that module integrates a phonon density of states for a crystal, this one sums
over a *discrete* mode list and adds the ideal-gas translational/rotational
partition functions that a molecule needs but a solid does not.

Three statistical-mechanical models are provided, mirroring ASE's
``IdealGasThermo`` and ``HarmonicThermo`` formulations term for term:

* :func:`ideal_gas_thermo` — a gas molecule: rigid-rotor / harmonic-oscillator
  (RRHO) with the Sackur-Tetrode translational entropy, rotational entropy
  (linear vs nonlinear, symmetry number ``σ``), harmonic vibrations, and an
  electronic spin degeneracy. Returns ZPE, U(T), H(T), S(T), G(T, p).
* :func:`harmonic_thermo` — an adsorbate: every degree of freedom is a harmonic
  vibration (the frustrated translations/rotations of the free molecule become
  low-frequency modes). Returns ZPE, U(T), S(T), and the Helmholtz free energy.
* :func:`adsorption_free_energy` — assembles ``ΔG_ads(T) = G[slab+ads] −
  G[slab] − ν·G[gas]`` from the two above, with a ``stoich_gas`` coefficient
  ``ν`` so a half-molecule reference (½ H₂ for adsorbed H) drops straight in.

On top of that, :func:`hydrogen_reference` and :func:`electrode_potential_shift`
implement the CHE: an adsorbed proton is referenced to ½ H₂(g), and a
proton-coupled electron-transfer step picks up ``+n·(eU + k_BT·ln10·pH)`` when
the electrode is held at potential ``U`` and the electrolyte at ``pH`` (Nørskov
et al., *J. Phys. Chem. B* **108**, 17886 (2004)). This is where the electrode
potential enters the free energy, and it composes with gradwave's constant-μ ESM
electrode for a full electrocatalytic ΔG(U, pH).

Everything is expressed in torch and stays differentiable in the energies, the
frequencies, and the electrode potential, so ``dΔG_ads/dU`` (Butler-Volmer
slope) and ``dΔG_ads/dω`` (Sabatier-optimal-binding fitting) come from autograd.
Frequencies enter as *energies in eV* (``ħω``, matching ASE's ``vib_energies``);
:func:`cm1_to_ev` converts a phonon/Hessian frequency list in cm⁻¹. Masses are
amu, moments of inertia amu·Å², temperatures K, pressures Pa, energies eV.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import torch
from torch import Tensor

from gradwave.constants import CM1_TO_EV, KB_EV
from gradwave.dtypes import RDTYPE as _RDTYPE

# --- physical constants in SI (CODATA 2018, consistent with gradwave.constants) ---
_KB_J = 1.380649e-23  # Boltzmann constant, J/K
_H_PLANCK = 6.62607015e-34  # Planck constant, J·s
_AMU_KG = 1.66053906660e-27  # atomic mass unit, kg
_P_REF = 1.0e5  # reference pressure, Pa (1 bar) — the standard state
_LN10 = math.log(10.0)

_Geometry = Literal["monatomic", "linear", "nonlinear"]


def cm1_to_ev(freqs_cm: Any) -> Tensor:
    """Convert a frequency list in cm⁻¹ to mode energies ħω in eV.

    The partial-Hessian / phonon machinery reports frequencies in cm⁻¹; the
    thermo functions here take mode *energies* in eV (ASE's ``vib_energies``
    convention). Imaginary modes carried as negative cm⁻¹ stay negative in eV,
    so :func:`clean_vib_energies` can flag them downstream.
    """
    return _as_tensor(freqs_cm) * CM1_TO_EV


def ev_to_cm1(energies_ev: Any) -> Tensor:
    """Inverse of :func:`cm1_to_ev`: mode energies in eV back to cm⁻¹."""
    return _as_tensor(energies_ev) / CM1_TO_EV


def _as_tensor(x: Any) -> Tensor:
    """Coerce a scalar / sequence / tensor to a 1-D-or-scalar float64 tensor.

    An existing tensor is returned untouched (grad history and device intact) so
    the thermo path stays differentiable when the caller passes a tensor that
    requires grad; a Python scalar or list becomes a fresh float64 leaf.
    """
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x, dtype=_RDTYPE)


def clean_vib_energies(
    vib_energies_ev: Any, ignore_imag_modes: bool = False
) -> tuple[Tensor, int]:
    """Split a mode-energy list into its real part and an imaginary-mode count.

    A harmonic minimum has no imaginary modes and a first-order transition state
    has exactly one; this module follows the phonon-DOS convention of
    :mod:`gradwave.postscf.thermo`, where an imaginary mode is carried as a
    *negative* energy (negative curvature of the PES). Non-positive entries are
    therefore the imaginary modes.

    With ``ignore_imag_modes`` the non-positive modes are dropped (with a
    warning) and the surviving positive modes are returned; otherwise their
    presence raises, so a caller that expects a clean minimum finds out. The
    returned tensor keeps grad history via a boolean mask (no in-place edit).
    """
    e = _as_tensor(vib_energies_ev).reshape(-1)
    is_real = e > 0.0
    n_imag = int((~is_real).sum().item())
    if ignore_imag_modes:
        if n_imag > 0:
            warnings.warn(
                f"{n_imag} imaginary vibrational mode(s) removed from the "
                "thermochemistry.",
                UserWarning,
                stacklevel=2,
            )
        return e[is_real], n_imag
    if n_imag > 0:
        raise ValueError(
            f"{n_imag} imaginary vibrational mode(s) present "
            "(non-positive energies); pass ignore_imag_modes=True to drop them, "
            "or use from_transition_state semantics for a saddle point."
        )
    return e, n_imag


# --- vibrational (harmonic-oscillator) building blocks ---------------------
def zero_point_energy(vib_energies_ev: Any) -> Tensor:
    """Zero-point energy ½ Σ ħωᵢ in eV over a mode-energy list.

    Each harmonic oscillator contributes ½ħω at T=0, independent of temperature.
    Only the real (positive) modes are summed; feed a cleaned list.
    """
    e = _as_tensor(vib_energies_ev)
    e = e[e > 0.0]
    return 0.5 * e.sum()


def vib_internal_energy(vib_energies_ev: Any, temperature: float) -> Tensor:
    """Thermal vibrational internal energy Σ ħω/(e^{ħω/kT}−1) in eV (ΔU, 0→T).

    This is the occupation term only; the zero-point energy is added separately
    by the callers so ZPE and the thermal part can be reported apart. At T→0 the
    Bose factor sends every term to zero.
    """
    e = _as_tensor(vib_energies_ev)
    e = e[e > 0.0]
    if temperature <= 0.0 or e.numel() == 0:
        return torch.zeros((), dtype=_RDTYPE)
    x = e / (KB_EV * temperature)
    return (e / torch.expm1(x)).sum()


def vib_entropy(vib_energies_ev: Any, temperature: float) -> Tensor:
    """Vibrational entropy S_vib = k_B Σ [x/(e^x−1) − ln(1−e^{−x})] in eV/K.

    With x = ħω/k_BT, this is the harmonic-oscillator entropy, the Legendre
    partner of the internal-energy term above. It is non-negative and rises with
    T; at T→0 every mode is frozen and S_vib → 0.
    """
    e = _as_tensor(vib_energies_ev)
    e = e[e > 0.0]
    if temperature <= 0.0 or e.numel() == 0:
        return torch.zeros((), dtype=_RDTYPE)
    x = e / (KB_EV * temperature)
    occ = x / torch.expm1(x)
    log_term = torch.log(-torch.expm1(-x))
    return KB_EV * (occ - log_term).sum()


# --- ideal-gas rigid-rotor / harmonic-oscillator thermo --------------------
def _translational_entropy(
    masses_amu: Any, temperature: float, pressure: float
) -> Tensor:
    """Sackur-Tetrode translational entropy in eV/K at (T, p).

    S_t = k_B[ln((2π m k_BT/h²)^{3/2} · k_BT/p) + 5/2], the entropy of a
    structureless ideal gas of molecular mass m at pressure p. The bracket is
    the log of the thermal number density times the quantum concentration; it is
    evaluated in SI and only the outer k_B carries the eV/K unit.
    """
    mass_kg = _as_tensor(masses_amu).sum() * _AMU_KG
    arg = (2.0 * math.pi * mass_kg * _KB_J * temperature / _H_PLANCK**2) ** 1.5
    arg = arg * _KB_J * temperature / pressure
    return KB_EV * (torch.log(arg) + 2.5)


def _rotational_entropy(
    geometry: _Geometry,
    moments_of_inertia_amu_a2: Any,
    symmetrynumber: int,
    temperature: float,
) -> Tensor:
    """Rigid-rotor rotational entropy in eV/K for a linear or nonlinear molecule.

    Nonlinear: S_r = k_B[ln(√(π ∏Iₖ)/σ · (8π²k_BT/h²)^{3/2}) + 3/2] over the
    three principal moments. Linear: S_r = k_B[ln(8π²I k_BT/(σh²)) + 1] with the
    single non-zero moment. Monatomic returns zero. Moments enter in amu·Å² and
    are converted to kg·m² (Å² = 10⁻²⁰ m²) before the SI log.
    """
    if geometry == "monatomic":
        return torch.zeros((), dtype=_RDTYPE)
    inertia_kg_m2 = _as_tensor(moments_of_inertia_amu_a2) * _AMU_KG / 1.0e20
    if geometry == "nonlinear":
        pref = torch.sqrt(math.pi * torch.prod(inertia_kg_m2)) / symmetrynumber
        pref = pref * (8.0 * math.pi**2 * _KB_J * temperature / _H_PLANCK**2) ** 1.5
        return KB_EV * (torch.log(pref) + 1.5)
    if geometry == "linear":
        inertia = inertia_kg_m2.max()
        arg = (
            8.0 * math.pi**2 * inertia * _KB_J * temperature
            / (symmetrynumber * _H_PLANCK**2)
        )
        return KB_EV * (torch.log(arg) + 1.0)
    raise ValueError(f"Unknown geometry: {geometry!r}")


def ideal_gas_thermo(
    vib_energies_ev: Any,
    geometry: _Geometry,
    *,
    temperature: float,
    potentialenergy: Any = 0.0,
    masses_amu: Any = None,
    moments_of_inertia_amu_a2: Any = None,
    symmetrynumber: int | None = None,
    spin: float = 0.0,
    pressure: float = _P_REF,
    ignore_imag_modes: bool = False,
) -> dict[str, Any]:
    """Ideal-gas RRHO thermodynamics of a molecule at temperature ``T``.

    Mirrors ASE's ``IdealGasThermo``: the free energy factorizes into
    translational (Sackur-Tetrode), rotational (rigid rotor), vibrational
    (harmonic), and electronic (spin-degeneracy) parts. Returns a dict with the
    zero-point energy, internal energy ``U``, enthalpy ``H``, entropy ``S``, and
    Gibbs free energy ``G`` (all torch scalars, eV except ``S`` in eV/K), plus an
    ``entropy_terms`` breakdown and the imaginary-mode count.

    Parameters mirror the physics: ``geometry`` is ``'monatomic'``, ``'linear'``,
    or ``'nonlinear'``; ``symmetrynumber`` is the rotational symmetry number σ
    (required unless monatomic); ``spin`` is the total electronic spin S so the
    degeneracy is 2S+1; ``pressure`` is the partial pressure in Pa.
    ``potentialenergy`` is the DFT total energy of the molecule; leave it 0 to
    get the free-energy *correction* to add to a bare energy.
    """
    e_real, n_imag = clean_vib_energies(vib_energies_ev, ignore_imag_modes)
    e_pot = _as_tensor(potentialenergy)
    zpe = zero_point_energy(e_real)

    if masses_amu is None:
        raise ValueError("masses_amu is required for an ideal gas "
                         "(the translational entropy needs the total mass).")
    if geometry in ("linear", "nonlinear"):
        if symmetrynumber is None:
            raise ValueError("symmetrynumber is required for a linear/nonlinear gas.")
        if moments_of_inertia_amu_a2 is None:
            raise ValueError("moments_of_inertia_amu_a2 is required for a "
                             "linear/nonlinear gas (the rotational entropy needs it).")

    # --- internal energy U = E + ZPE + Cv_trans·T + Cv_rot·T + ΔU_vib ---
    cv_trans_t = 1.5 * KB_EV * temperature
    if geometry == "nonlinear":
        cv_rot_t = 1.5 * KB_EV * temperature
    elif geometry == "linear":
        cv_rot_t = 1.0 * KB_EV * temperature
    else:
        cv_rot_t = 0.0
    du_vib = vib_internal_energy(e_real, temperature)
    internal_energy = e_pot + zpe + cv_trans_t + cv_rot_t + du_vib

    # --- enthalpy H = U + k_BT (ideal-gas Cv→Cp correction) ---
    enthalpy = internal_energy + KB_EV * temperature

    # --- entropy S = S_trans + S_rot + S_elec + S_vib + S_pressure ---
    s_trans = _translational_entropy(masses_amu, temperature, _P_REF)
    # _rotational_entropy already returns 0 for a monatomic gas, so no guard needed.
    s_rot = _rotational_entropy(
        geometry, moments_of_inertia_amu_a2, symmetrynumber or 1, temperature
    )
    s_elec = KB_EV * math.log(2.0 * spin + 1.0)
    s_vib = vib_entropy(e_real, temperature)
    # Pressure correction, 1 bar → p, folded into S_trans's standard state.
    s_pressure = -KB_EV * math.log(pressure / _P_REF)
    entropy = s_trans + s_rot + s_elec + s_vib + s_pressure

    gibbs = enthalpy - temperature * entropy
    return {
        "zero_point_energy": zpe,
        "internal_energy": internal_energy,
        "enthalpy": enthalpy,
        "entropy": entropy,
        "gibbs_energy": gibbs,
        "entropy_terms": {
            "translational": s_trans,
            "rotational": s_rot,
            "electronic": torch.as_tensor(s_elec, dtype=_RDTYPE),
            "vibrational": s_vib,
            "pressure": torch.as_tensor(s_pressure, dtype=_RDTYPE),
        },
        "n_imag": n_imag,
        "temperature": temperature,
        "pressure": pressure,
    }


# --- harmonic (adsorbate) thermo -------------------------------------------
def harmonic_thermo(
    vib_energies_ev: Any,
    *,
    temperature: float,
    potentialenergy: Any = 0.0,
    ignore_imag_modes: bool = False,
) -> dict[str, Any]:
    """Harmonic thermodynamics of an adsorbate at temperature ``T``.

    Mirrors ASE's ``HarmonicThermo``: every degree of freedom is a quantum
    harmonic oscillator (the frustrated translations and rotations of the free
    molecule show up as low-frequency modes in the list, so there is no separate
    translational or rotational term). Returns a dict with the zero-point energy,
    internal energy ``U = E + ZPE + ΔU_vib``, entropy ``S = S_vib``, and the
    Helmholtz free energy ``F = U − T·S`` (torch scalars), plus ``n_imag``.

    A true minimum has ``n_imag == 0``; a first-order saddle has one imaginary
    mode. Pass ``ignore_imag_modes=True`` to drop imaginary modes (e.g. to get
    the thermal correction at a transition state after removing the reaction
    coordinate), otherwise their presence raises.
    """
    e_real, n_imag = clean_vib_energies(vib_energies_ev, ignore_imag_modes)
    e_pot = _as_tensor(potentialenergy)
    zpe = zero_point_energy(e_real)
    du_vib = vib_internal_energy(e_real, temperature)
    entropy = vib_entropy(e_real, temperature)
    internal_energy = e_pot + zpe + du_vib
    helmholtz = internal_energy - temperature * entropy
    return {
        "zero_point_energy": zpe,
        "internal_energy": internal_energy,
        "entropy": entropy,
        "helmholtz_energy": helmholtz,
        "n_imag": n_imag,
        "temperature": temperature,
    }


# --- ΔG_ads assembly -------------------------------------------------------
def adsorption_free_energy(
    *,
    energy_slab_ads: Any,
    energy_slab: Any,
    energy_gas: Any,
    temperature: float,
    ads_vib_energies_ev: Any,
    gas_vib_energies_ev: Any,
    gas_geometry: _Geometry,
    gas_masses_amu: Any = None,
    gas_moments_of_inertia_amu_a2: Any = None,
    gas_symmetrynumber: int | None = None,
    gas_spin: float = 0.0,
    slab_vib_energies_ev: Any = None,
    stoich_gas: float = 1.0,
    pressure: float = _P_REF,
    ignore_imag_modes: bool = False,
) -> dict[str, Any]:
    """Free energy of adsorption ΔG_ads(T) from DFT energies + frequencies.

    Assembles ``ΔG_ads = G[slab+ads] − G[slab] − ν·G[gas]`` at temperature ``T``,
    where the adsorbed complex and the (optionally vibrating) slab are treated as
    harmonic solids and the gas reference is a full ideal-gas RRHO molecule. The
    stoichiometric coefficient ``ν = stoich_gas`` scales the gas term so a
    half-molecule reference drops in directly (``ν = 0.5`` for ½ H₂ → adsorbed
    H, the CHE reference).

    ``G[slab+ads]`` and ``G[slab]`` use the Helmholtz free energy of
    :func:`harmonic_thermo` (a slab is condensed, so its ``pV`` term is
    negligible and ``G ≈ F``); with ``slab_vib_energies_ev=None`` the clean slab
    contributes only its DFT energy (the frozen-substrate approximation).
    ``G[gas]`` is the Gibbs free energy of :func:`ideal_gas_thermo`.

    Returns a dict with ``delta_g`` (a torch scalar, differentiable in every
    energy, frequency, and — once composed with :func:`electrode_potential_shift`
    — the electrode potential), the three component free energies, and the
    adsorbate imaginary-mode count ``n_imag_ads``.
    """
    g_slab_ads = harmonic_thermo(
        ads_vib_energies_ev,
        temperature=temperature,
        potentialenergy=energy_slab_ads,
        ignore_imag_modes=ignore_imag_modes,
    )
    if slab_vib_energies_ev is None:
        g_slab = _as_tensor(energy_slab)
        n_imag_slab = 0
    else:
        slab_res = harmonic_thermo(
            slab_vib_energies_ev,
            temperature=temperature,
            potentialenergy=energy_slab,
            ignore_imag_modes=ignore_imag_modes,
        )
        g_slab = slab_res["helmholtz_energy"]
        n_imag_slab = slab_res["n_imag"]
    g_gas = ideal_gas_thermo(
        gas_vib_energies_ev,
        gas_geometry,
        temperature=temperature,
        potentialenergy=energy_gas,
        masses_amu=gas_masses_amu,
        moments_of_inertia_amu_a2=gas_moments_of_inertia_amu_a2,
        symmetrynumber=gas_symmetrynumber,
        spin=gas_spin,
        pressure=pressure,
        ignore_imag_modes=ignore_imag_modes,
    )
    delta_g = (
        g_slab_ads["helmholtz_energy"]
        - g_slab
        - stoich_gas * g_gas["gibbs_energy"]
    )
    return {
        "delta_g": delta_g,
        "g_slab_ads": g_slab_ads["helmholtz_energy"],
        "g_slab": g_slab,
        "g_gas": g_gas["gibbs_energy"],
        "stoich_gas": stoich_gas,
        "n_imag_ads": g_slab_ads["n_imag"],
        "n_imag_slab": n_imag_slab,
        "temperature": temperature,
    }


# --- computational hydrogen electrode (CHE) --------------------------------
def hydrogen_reference(gibbs_h2: Any) -> Tensor:
    """Chemical potential of an adsorbed H reference, ½·G(H₂(g)), in eV.

    The computational hydrogen electrode references a proton-electron pair
    ``(H⁺ + e⁻)`` to half a gas-phase H₂ molecule at U = 0 V vs RHE and pH 0, so
    ``μ(H⁺ + e⁻) = ½ G(H₂)`` there. Feed this as the gas term (or use
    ``stoich_gas=0.5`` with H₂ in :func:`adsorption_free_energy`).
    """
    return 0.5 * _as_tensor(gibbs_h2)


def electrode_potential_shift(
    delta_g: Any,
    *,
    potential_v: Any = 0.0,
    ph: float = 0.0,
    temperature: float = 298.15,
    n_electrons: int = 1,
) -> Tensor:
    """Shift a CHE reaction free energy to electrode potential ``U`` and ``pH``.

    For a proton-coupled electron-transfer step the free energy of each
    transferred ``(H⁺ + e⁻)`` pair moves by ``eU + k_BT·ln10·pH`` relative to the
    ½ H₂ reference (Nørskov et al. 2004), so

        ΔG(U, pH) = ΔG_CHE + n·(eU + k_BT·ln10·pH),

    with ``n = n_electrons`` the number of pairs in the step and ``U`` in volts
    vs RHE (numerically ``eU`` in eV equals ``U`` in V). The result stays
    differentiable in ``potential_v``: ``d(ΔG)/dU = n``, the ideal Nernstian /
    Butler-Volmer slope, comes straight from autograd, and this term is what
    composes with gradwave's constant-μ ESM electrode. Pass ``potential_v`` as a
    tensor with ``requires_grad=True`` to read the slope back.
    """
    dg = _as_tensor(delta_g)
    u = _as_tensor(potential_v)
    return dg + n_electrons * (u + KB_EV * temperature * _LN10 * ph)
