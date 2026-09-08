"""Ergonomic free-energy thermochemistry helpers (Layer C).

Thin ASE-aware wrappers over :mod:`gradwave.postscf.adsorbate_thermo`. The
postscf module is deliberately ASE-free and takes masses / moments of inertia as
plain arrays so it stays a differentiable numeric leaf; this layer reads those
straight off an ``ase.Atoms`` gas molecule (mass, principal moments of inertia,
and — if not given — the linear/nonlinear geometry) so a caller with an Atoms
object does not assemble them by hand. The heavy lifting, and the full
docstrings for the physics, live in the postscf module; the primitives are
re-exported here so ``from gradwave.api import ...`` reaches them.

These consume the adsorbate frequencies produced by the partial (active-atom)
Hessian and turn them into ΔG_ads(T) for microkinetics and the computational
hydrogen electrode; :func:`gradwave.postscf.adsorbate_thermo.electrode_potential_shift`
adds the ΔG(U, pH) term for electrocatalysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from gradwave.inputs import Input
from gradwave.postscf.adsorbate_thermo import (
    adsorption_free_energy as adsorption_free_energy,
)
from gradwave.postscf.adsorbate_thermo import (
    cm1_to_ev as cm1_to_ev,
)
from gradwave.postscf.adsorbate_thermo import (
    electrode_potential_shift as electrode_potential_shift,
)
from gradwave.postscf.adsorbate_thermo import (
    harmonic_thermo as harmonic_thermo,
)
from gradwave.postscf.adsorbate_thermo import (
    hydrogen_reference as hydrogen_reference,
)
from gradwave.postscf.adsorbate_thermo import (
    ideal_gas_thermo as ideal_gas_thermo,
)

if TYPE_CHECKING:
    from ase import Atoms

_Geometry = Literal["monatomic", "linear", "nonlinear"]


def _detect_geometry(atoms: Atoms) -> _Geometry:
    """Classify a molecule as monatomic, linear, or nonlinear from its moments.

    A single atom is monatomic; a molecule with one (near-)zero principal moment
    of inertia is linear; otherwise nonlinear. The zero test is relative to the
    largest moment so it is scale-free.
    """
    n = len(atoms)
    if n == 1:
        return "monatomic"
    moments = atoms.get_moments_of_inertia()
    max_moment = float(max(moments))
    if max_moment <= 0.0:
        return "monatomic"
    n_zero = int(sum(1 for m in moments if abs(m) < 1e-6 * max_moment))
    return "linear" if n_zero >= 1 else "nonlinear"


def molecule_ideal_gas_thermo(
    atoms: Atoms,
    vib_energies_ev: Any,
    *,
    temperature: float,
    symmetrynumber: int | None = None,
    spin: float = 0.0,
    geometry: str | None = None,
    potentialenergy: Any = 0.0,
    pressure: float = 1.0e5,
    ignore_imag_modes: bool = False,
) -> dict[str, Any]:
    """Ideal-gas RRHO thermo of a gas molecule given as an ``ase.Atoms``.

    Reads the molecular mass and principal moments of inertia off ``atoms`` and
    (unless ``geometry`` is passed) detects linear vs nonlinear, then defers to
    :func:`gradwave.postscf.adsorbate_thermo.ideal_gas_thermo`. Frequencies enter
    as mode energies in eV (use :func:`cm1_to_ev` for a cm⁻¹ list); the return
    dict is that function's, with ZPE / U / H / S / G.
    """
    geom = cast("_Geometry", geometry) if geometry else _detect_geometry(atoms)
    return ideal_gas_thermo(
        vib_energies_ev,
        geom,
        temperature=temperature,
        potentialenergy=potentialenergy,
        masses_amu=atoms.get_masses(),
        moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        symmetrynumber=symmetrynumber,
        spin=spin,
        pressure=pressure,
        ignore_imag_modes=ignore_imag_modes,
    )


def adsorption_free_energy_from_atoms(
    *,
    energy_slab_ads: Any,
    energy_slab: Any,
    energy_gas: Any,
    temperature: float,
    ads_vib_energies_ev: Any,
    gas_vib_energies_ev: Any,
    gas_atoms: Atoms,
    gas_symmetrynumber: int | None = None,
    gas_spin: float = 0.0,
    gas_geometry: str | None = None,
    slab_vib_energies_ev: Any = None,
    stoich_gas: float = 1.0,
    pressure: float = 1.0e5,
    ignore_imag_modes: bool = False,
) -> dict[str, Any]:
    """ΔG_ads(T) with the gas reference supplied as an ``ase.Atoms``.

    Convenience over
    :func:`gradwave.postscf.adsorbate_thermo.adsorption_free_energy` that pulls
    the gas molecule's mass, moments of inertia, and (optionally) geometry from
    ``gas_atoms``. Returns that function's dict, with a differentiable
    ``delta_g``.
    """
    geom = (
        cast("_Geometry", gas_geometry)
        if gas_geometry
        else _detect_geometry(gas_atoms)
    )
    return adsorption_free_energy(
        energy_slab_ads=energy_slab_ads,
        energy_slab=energy_slab,
        energy_gas=energy_gas,
        temperature=temperature,
        ads_vib_energies_ev=ads_vib_energies_ev,
        gas_vib_energies_ev=gas_vib_energies_ev,
        gas_geometry=geom,
        gas_masses_amu=gas_atoms.get_masses(),
        gas_moments_of_inertia_amu_a2=gas_atoms.get_moments_of_inertia(),
        gas_symmetrynumber=gas_symmetrynumber,
        gas_spin=gas_spin,
        slab_vib_energies_ev=slab_vib_energies_ev,
        stoich_gas=stoich_gas,
        pressure=pressure,
        ignore_imag_modes=ignore_imag_modes,
    )


def _f(x: Any) -> float:
    """A torch scalar / python number → plain float for the JSON summary."""
    return float(x.item() if hasattr(x, "item") else x)


def run_thermochem(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Free-energy thermochemistry (task: thermochem) — no SCF.

    Turns the DFT energies and harmonic frequency lists carried in
    ``inp.thermochem`` into temperature-dependent free energies via
    :mod:`gradwave.postscf.adsorbate_thermo`. The three modes (``adsorption`` /
    ``ideal_gas`` / ``harmonic``) map to :func:`adsorption_free_energy_from_atoms`,
    :func:`molecule_ideal_gas_thermo`, and :func:`harmonic_thermo`; the top-level
    ``structure`` is the gas / molecule whose mass and moments of inertia feed the
    ideal-gas rotational and translational entropy. Returns the ``thermochem``
    summary block (all energies eV, entropy eV/K, temperature K)."""
    tc = inp.thermochem
    t = tc.temperature
    block: dict[str, Any] = {"mode": tc.mode, "temperature_K": t}

    if tc.mode == "harmonic":
        res = harmonic_thermo(
            cm1_to_ev(tc.freqs_cm), temperature=t, potentialenergy=tc.energy,
            ignore_imag_modes=tc.ignore_imag_modes)
        block.update({
            "zero_point_energy_eV": _f(res["zero_point_energy"]),
            "internal_energy_eV": _f(res["internal_energy"]),
            "entropy_eV_per_K": _f(res["entropy"]),
            "helmholtz_energy_eV": _f(res["helmholtz_energy"]),
            "free_energy_eV": _f(res["helmholtz_energy"]),
            "n_imag": int(res["n_imag"]),
        })
    elif tc.mode == "ideal_gas":
        res = molecule_ideal_gas_thermo(
            inp.atoms, cm1_to_ev(tc.freqs_cm), temperature=t,
            symmetrynumber=tc.symmetrynumber, spin=tc.spin,
            geometry=tc.geometry, potentialenergy=tc.energy,
            pressure=tc.pressure, ignore_imag_modes=tc.ignore_imag_modes)
        block.update({
            "pressure_Pa": tc.pressure,
            "geometry": tc.geometry or _detect_geometry(inp.atoms),
            "zero_point_energy_eV": _f(res["zero_point_energy"]),
            "internal_energy_eV": _f(res["internal_energy"]),
            "enthalpy_eV": _f(res["enthalpy"]),
            "entropy_eV_per_K": _f(res["entropy"]),
            "gibbs_energy_eV": _f(res["gibbs_energy"]),
            "free_energy_eV": _f(res["gibbs_energy"]),
            "n_imag": int(res["n_imag"]),
        })
    else:  # adsorption
        res = adsorption_free_energy_from_atoms(
            energy_slab_ads=tc.energy_slab_ads, energy_slab=tc.energy_slab,
            energy_gas=tc.energy_gas, temperature=t,
            ads_vib_energies_ev=cm1_to_ev(tc.ads_freqs_cm),
            gas_vib_energies_ev=cm1_to_ev(tc.gas_freqs_cm),
            gas_atoms=inp.atoms, gas_symmetrynumber=tc.gas_symmetrynumber,
            gas_spin=tc.gas_spin, gas_geometry=tc.gas_geometry,
            slab_vib_energies_ev=(None if tc.slab_freqs_cm is None
                                  else cm1_to_ev(tc.slab_freqs_cm)),
            stoich_gas=tc.stoich_gas, pressure=tc.pressure,
            ignore_imag_modes=tc.ignore_imag_modes)
        block.update({
            "pressure_Pa": tc.pressure,
            "stoich_gas": tc.stoich_gas,
            "delta_g_eV": _f(res["delta_g"]),
            "free_energy_eV": _f(res["delta_g"]),
            "g_slab_ads_eV": _f(res["g_slab_ads"]),
            "g_slab_eV": _f(res["g_slab"]),
            "g_gas_eV": _f(res["g_gas"]),
            "n_imag_ads": int(res["n_imag_ads"]),
            "n_imag_slab": int(res["n_imag_slab"]),
        })

    # computational-hydrogen-electrode shift ΔG(U, pH), applied to the reported
    # free energy (adsorption ΔG or a single-species G) when a potential is set.
    if tc.electrode_potential_v is not None:
        shifted = electrode_potential_shift(
            block["free_energy_eV"], potential_v=tc.electrode_potential_v,
            ph=tc.ph, temperature=t, n_electrons=tc.n_electrons)
        block["electrode"] = {
            "potential_v": tc.electrode_potential_v,
            "ph": tc.ph,
            "n_electrons": tc.n_electrons,
            "free_energy_shifted_eV": _f(shifted),
        }

    if verbose:
        print(f"thermochem ({tc.mode}): G = {block['free_energy_eV']:+.6f} eV "
              f"at {t:.2f} K", flush=True)
    return block
