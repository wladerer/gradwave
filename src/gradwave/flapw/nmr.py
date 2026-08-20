"""Solid-state NMR observables from the FLAPW electric field gradient.

The quadrupolar coupling of a nucleus with quadrupole moment ``Q`` in an electric field gradient
``V_zz`` is ``C_Q = e Q V_zz / h``. The FLAPW ``efg_tensor`` returns ``V_zz`` as the curvature of
the Coulomb potential energy (eV/Å²); converting to the electrostatic-potential gradient and to
SI units gives

    C_Q [MHz] = 234.965 · Q[barn] · V_zz[a.u.] = 2.4180 · Q[barn] · V_zz[eV/Å²],

where ``1 a.u.`` of EFG is ``9.7174e21 V/m²`` and ``1 eV/Å² = 1e20 V/m²`` of potential gradient.
The asymmetry ``η = |V_xx−V_yy|/|V_zz|`` (``0 ≤ η ≤ 1``) is the second measured quadrupolar
parameter. The observed powder splitting is ``ν_Q = 3 C_Q / [2 I (2I−1)]`` for spin ``I``.

Nuclear quadrupole moments (barn) and spins from the IAEA/Stone tabulation; the sign of ``C_Q``
follows the sign of ``Q·V_zz`` but experiments report ``|C_Q|``.
"""

from __future__ import annotations

# isotope -> (quadrupole moment Q [barn], nuclear spin I)
NUCLEAR_Q: dict[str, tuple[float, float]] = {
    "2H": (0.00286, 1.0),
    "7Li": (-0.04010, 1.5),
    "11B": (0.04059, 1.5),
    "14N": (0.02044, 1.0),
    "17O": (-0.02558, 2.5),
    "23Na": (0.10400, 1.5),
    "27Al": (0.14660, 2.5),
    "47Ti": (0.30200, 2.5),
    "49Ti": (0.24700, 3.5),
    "63Cu": (-0.22000, 1.5),
}

_MHZ_PER_BARN_EV_A2 = 2.4180        # C_Q[MHz] = k · Q[barn] · V_zz[eV/Å²]


def quadrupolar_coupling(v_zz: float, eta: float, isotope: str) -> dict:
    """The NMR quadrupolar parameters for ``isotope`` in an EFG with principal component ``v_zz``
    (eV/Å²) and asymmetry ``eta``. Returns ``C_Q`` (MHz), ``|C_Q|``, ``η``, the quadrupolar
    frequency ``ν_Q`` (MHz), the nuclear spin, and ``Q`` (barn)."""
    if isotope not in NUCLEAR_Q:
        raise KeyError(f"no quadrupole moment for {isotope}; known: {sorted(NUCLEAR_Q)}")
    q_barn, spin = NUCLEAR_Q[isotope]
    c_q = _MHZ_PER_BARN_EV_A2 * q_barn * v_zz
    nu_q = 3.0 * abs(c_q) / (2.0 * spin * (2.0 * spin - 1.0)) if spin > 0.5 else 0.0
    return {"isotope": isotope, "C_Q_MHz": c_q, "abs_C_Q_MHz": abs(c_q), "eta": eta,
            "nu_Q_MHz": nu_q, "spin": spin, "Q_barn": q_barn}
