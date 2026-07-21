"""Periodic-table Δ-gauge cases: PseudoDojo NC-SR (v0.4, PBE, standard) vs the
WIEN2k all-electron reference.

Protocol (Lejaeghere et al., Science 351, aad3000, 2016): E(V) at seven volumes
spanning 94-106% of the ALL-ELECTRON equilibrium volume, third-order
Birch-Murnaghan fit, Δ = RMS difference of the fitted curves (each shifted to
its own minimum) over the ±6% window around the average V0, per atom.

Two comparison axes (see docs/manual/wisdom.md "Process and validation"):
  Δ vs WIEN2k  — mixes pseudization with implementation; the fair target is the
                 pseudo's own published Δ (`dfact`, from the PseudoDojo table).
  Δ vs QE      — same pseudo/cutoff/mesh, isolates the implementation (~0.01
                 meV/atom scale for norm-conserving, per benchmarks/delta_factor).

`ecut` is 2× the PseudoDojo "high" hint (Ha→Ry); ecutrho defaults to 4×ecut for
norm-conserving. `dfact` is PseudoDojo's own Δ-factor (meV/atom) for the standard
pseudo — the three-way cross-check the gradwave Δ_wien2k should track.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lattices import geometry as _geom  # noqa: E402
from lattices import natoms as _natoms  # noqa: E402

RY = 13.605693122994

# WIEN2k v13.1 (calcDelta 3.0): V0 (Å³/atom), B0 (GPa), B1. Authoritative EOS
# reference for all 71 elements; the cubic subset below is what V0-fixed
# geometry can address.
WIEN2K = {
    "H": (17.3883, 10.284, 2.71), "He": (17.7708, 0.847, 7.71),
    "Li": (20.2191, 13.839, 3.34), "Be": (7.9099, 122.903, 3.04),
    "B": (7.2405, 237.290, 3.47), "C": (11.6366, 208.991, 3.58),
    "N": (28.8848, 54.2195, 3.7244), "O": (18.5590, 51.378, 3.89),
    "F": (19.1666, 34.325, 3.93), "Ne": (24.2492, 1.406, 14.44),
    "Na": (37.4686, 7.472, 3.77), "Mg": (22.9355, 35.933, 4.07),
    "Al": (16.4796, 78.077, 4.57), "Si": (20.4530, 88.545, 4.31),
    "P": (21.4709, 68.208, 4.35), "S": (17.1840, 83.407, 4.26),
    "Cl": (38.8889, 19.081, 4.34), "Ar": (52.3852, 0.743, 7.26),
    "K": (73.6793, 3.574, 4.59), "Ca": (42.1991, 17.114, 3.31),
    "Sc": (24.6196, 54.393, 3.42), "Ti": (17.3900, 112.213, 3.58),
    "V": (13.4520, 181.674, 3.75), "Cr": (11.7730, 183.899, 7.16),
    "Mn": (11.4473, 118.632, -0.21), "Fe": (11.3436, 197.652, 5.80),
    "Co": (10.8599, 217.295, 4.37), "Ni": (10.8876, 200.368, 5.00),
    "Cu": (11.9511, 141.335, 4.86), "Zn": (15.1820, 74.780, 5.26),
    "Ga": (20.3069, 49.223, 5.38), "Ge": (23.9148, 59.128, 4.99),
    "As": (22.5890, 68.285, 4.22), "Se": (29.7437, 47.070, 4.44),
    "Br": (39.4470, 22.415, 4.87), "Kr": (65.6576, 0.671, 9.86),
    "Rb": (90.8087, 2.787, 5.80), "Sr": (54.5272, 11.256, 3.49),
    "Y": (32.8442, 41.593, 3.02), "Zr": (23.3850, 93.684, 3.21),
    "Nb": (18.1368, 171.270, 3.55), "Mo": (15.7862, 258.928, 4.33),
    "Tc": (14.4366, 299.149, 4.46), "Ru": (13.7619, 312.502, 4.95),
    "Rh": (14.0396, 257.824, 5.32), "Pd": (15.3101, 168.629, 5.56),
    "Ag": (17.8471, 90.148, 5.42), "Cd": (22.6287, 46.403, 6.92),
    "In": (27.4710, 34.937, 4.78), "Sn": (36.8166, 36.030, 4.64),
    "Sb": (31.7296, 50.367, 4.52), "Te": (34.9765, 44.787, 4.69),
    "I": (50.2333, 18.654, 5.05), "Xe": (86.6814, 0.548, 6.34),
    "Cs": (117.080, 1.982, 2.14), "Ba": (63.1401, 8.677, 3.77),
    "Lu": (29.0544, 46.384, 2.94), "Hf": (22.5325, 107.004, 3.50),
    "Ta": (18.2856, 195.147, 3.71), "W": (16.1394, 301.622, 4.28),
    "Re": (14.9580, 362.850, 4.52), "Os": (14.2802, 397.259, 4.84),
    "Ir": (14.5004, 347.680, 5.18), "Pt": (15.6420, 248.711, 5.46),
    "Au": (17.9745, 139.109, 5.76), "Hg": (29.5220, 8.204, 8.87),
    "Tl": (31.3902, 26.865, 5.49), "Pb": (32.0028, 39.544, 4.53),
    "Bi": (36.9047, 42.630, 4.70), "Po": (37.5869, 45.458, 4.93),
    "Rn": (92.6852, 0.564, 8.62),
}

# element -> case. ecut in Ry (= 2× PseudoDojo high-hint Ha). Zval = valence
# electrons of the standard pseudo; nbands is derived with a metallic buffer.
# smear/width (Ry): metals gaussian; group-IV insulators none (Ge/Sn are PBE
# near-zero-gap, so a whisker of smearing keeps occupations well-defined).
# dfact = PseudoDojo published Δ-factor (meV/atom) for the standard pseudo.
def _case(struct, Zval, ecut, k, smear="gaussian", width=0.01, nspin=1,
          start_mag=None, dfact=None):
    return dict(struct=struct, Zval=Zval, ecut=ecut, kmesh=(k, k, k),
                smear=smear, width=width, nspin=nspin, start_mag=start_mag,
                dfact=dfact)


CASES = {
    # alkali (bcc). K's cell is large (a≈5.3 Å) so its BZ is small: 12³ there
    # matches the reciprocal-space k-density of the ~3.6 Å cells at 16³, and it
    # keeps the 54³×n_k batched apply under the 6 GB GPU ceiling.
    "Li": _case("bcc", 3, 82, 16, dfact=0.173),
    "Na": _case("bcc", 9, 96, 16, dfact=0.435),
    "K":  _case("bcc", 9, 86, 12, dfact=0.177),
    # alkaline earth (fcc); Sr cell a≈6.0 Å -> 12³ for the same k-density
    "Ca": _case("fcc", 10, 76, 16, dfact=0.063),
    "Sr": _case("fcc", 10, 80, 12, dfact=1.320),
    # simple metal (fcc)
    "Al": _case("fcc", 3, 52, 16, dfact=0.542),
    # group IV (diamond)
    "Si": _case("diamond", 4, 48, 8, smear="none", width=0.0, dfact=0.146),
    "Ge": _case("diamond", 14, 90, 8, width=0.002, dfact=0.485),
    "Sn": _case("diamond", 14, 84, 8, width=0.002, dfact=0.772),
    # group V refractory (bcc)
    "V":  _case("bcc", 13, 96, 16, dfact=1.337),
    "Nb": _case("bcc", 13, 98, 16, dfact=1.292),
    "Ta": _case("bcc", 13, 70, 16, dfact=0.739),
    # group VI refractory (bcc)
    "Mo": _case("bcc", 14, 92, 16, dfact=1.405),
    "W":  _case("bcc", 14, 82, 16, dfact=0.188),
    # noble (fcc)
    "Cu": _case("fcc", 19, 104, 16, dfact=0.533),
    "Ag": _case("fcc", 19, 94, 16, dfact=0.315),
    "Au": _case("fcc", 19, 88, 16, dfact=1.349),
    # Pt-group (fcc)
    "Pd": _case("fcc", 18, 98, 16, dfact=1.135),
    "Pt": _case("fcc", 18, 100, 16, dfact=0.632),
    "Ir": _case("fcc", 17, 80, 16, dfact=1.452),
    "Rh": _case("fcc", 17, 100, 16, dfact=2.564),
    # post-transition (fcc)
    "Pb": _case("fcc", 14, 68, 16, dfact=0.063),
    # magnetic: Fe/Ni ferromagnetic (johnson mixer, energy-gated). Seed high and
    # let the moment relax down (weak seeds collapse to the NM branch). Cr is
    # antiferromagnetic and needs a 2-atom cell — not the 1-atom bcc primitive
    # here — so it is left out of the FM path.
    "Fe": _case("bcc", 16, 106, 12, nspin=2, start_mag=[0.7], dfact=5.599),
    "Ni": _case("fcc", 18, 110, 12, nspin=2, start_mag=[0.5], dfact=1.065),
    "Cr": _case("bcc", 14, 110, 12, nspin=2, start_mag=[0.7], dfact=10.499),
}

# nonmagnetic set that phase 1 runs; magnetic set gated behind johnson-mixer work
NONMAG = [e for e, c in CASES.items() if c["nspin"] == 1]
MAG = [e for e, c in CASES.items() if c["nspin"] == 2]


def nbands(elem):
    """Occupied bands plus a metallic buffer."""
    c = CASES[elem]
    nelec = c["Zval"] * _natoms(c["struct"])
    if c["smear"] == "none":
        return None  # insulator: let setup_system fill valence exactly
    return int(nelec / 2 * 1.35) + 6


def geometry(elem, scale):
    c = CASES[elem]
    return _geom(c["struct"], elem, WIEN2K[elem][0], scale)
