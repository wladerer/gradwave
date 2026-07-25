"""Magnetic-mineral structures for the gradwave-vs-QE SCF benchmark.

Every mineral is returned as a single, explicit cell that is fed *identically*
to both codes: same lattice vectors, same atom count, same Cartesian
positions. QE does not auto-primitivize, so the cell we build here IS the cell
both codes integrate. For the collinear antiferromagnets we build the MAGNETIC
primitive cell (the smallest cell that accommodates the AFM order) so the
Shubnikov (collinear-magnetic) k-fold is exercised on the real ordering.

Each ``Mineral`` carries:
  cell      (3,3) lattice vectors, Angstrom
  positions (nat,3) Cartesian, Angstrom
  species   list[int] index into ``pseudos``/``symbols``
  symbols   per-species element symbol (for the QE ATOMIC_SPECIES card)
  pseudos   per-species UPF filename (the SAME file both codes read)
  magmoms   (nat,3) collinear moment directions (z-axis), 0 on non-magnetic
            sites -> passed to setup_system(magmoms=..., collinear_magnetic=True)
  start_mag per-atom starting moment fraction (sign = sublattice), for scf()
            and QE starting_magnetization
  kmesh     Monkhorst-Pack grid (unshifted, Gamma-centred)
  ecut_ry   plane-wave cutoff (Ry); ecutrho = rho_factor*ecut_ry
  degauss_ry gaussian smearing width (Ry)
  nspin     2 for the AFM cases, 1 for diamagnetic pyrite
  pseudo_kind 'NC' | 'USPP' (sets ecutrho factor and is reported)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# masses for the QE ATOMIC_SPECIES card
MASS = {"Ni": 58.693, "O": 15.999, "Fe": 55.845, "Cr": 51.996,
        "S": 32.06, "H": 1.008, "C": 12.011, "Mn": 54.938}


@dataclass
class Mineral:
    name: str
    cell: np.ndarray
    positions: np.ndarray
    species: list[int]
    symbols: list[str]
    pseudos: list[str]
    magmoms: np.ndarray
    start_mag: list[float]
    kmesh: tuple[int, int, int]
    ecut_ry: float
    degauss_ry: float
    nspin: int
    pseudo_kind: str
    afm: bool
    note: str = ""
    rho_factor: float = field(default=4.0)  # ecutrho = rho_factor*ecutwfc

    @property
    def nat(self) -> int:
        return len(self.species)


def _find_magnetic_primitive(cell, scaled, labels, symprec=1e-4):
    """Collapse a supercell to its magnetic primitive by treating each distinct
    (element, spin-sign) as a distinct spglib species. Returns (cell, scaled,
    labels) of the primitive. Requires spglib."""
    import spglib

    uniq = sorted(set(labels))
    num_of = {lb: i + 1 for i, lb in enumerate(uniq)}
    numbers = [num_of[lb] for lb in labels]
    prim = spglib.find_primitive((cell, scaled, numbers), symprec=symprec)
    if prim is None:
        raise RuntimeError("spglib.find_primitive failed")
    pcell, pscaled, pnums = prim
    inv = {v: k for k, v in num_of.items()}
    plabels = [inv[int(n)] for n in pnums]
    return np.asarray(pcell), np.asarray(pscaled), plabels


def nio(a: float = 4.17, ecut_ry: float = 70.0, degauss_ry: float = 0.01,
        kmesh=(6, 6, 6)) -> Mineral:
    """NiO rocksalt, type-II AFM along [111]. The magnetic primitive is the
    4-atom rhombohedral cell (2 Ni antiparallel + 2 O). Built by tiling the
    conventional cubic cell, assigning the type-II sign to each Ni by its
    (111) plane index s = round(x+y+z) [cubic units], then extracting the
    magnetic primitive."""
    # conventional cubic NiO, 2x2x2 so the doubled [111] period is periodic
    base_ni = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    base_o = base_ni + np.array([0.5, 0.0, 0.0])
    labels, cart = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                shift = np.array([i, j, k], float)
                for r in base_ni:
                    p = r + shift
                    s = int(round(p[0] + p[1] + p[2]))
                    labels.append("Ni+" if s % 2 == 0 else "Ni-")
                    cart.append(p)
                for r in base_o:
                    cart.append(r + shift)
                    labels.append("O")
    super_cell = np.diag([2 * a, 2 * a, 2 * a])
    scaled = np.array(cart) / 2.0  # fractional of the 2x2x2 cell
    pcell, pscaled, plabels = _find_magnetic_primitive(super_cell, scaled, labels)

    symbols = ["Ni", "O"]
    pseudos = ["Ni_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.2.upf"]
    species, magmoms, start_mag = [], [], []
    for lb, _fr in zip(plabels, pscaled, strict=True):
        if lb.startswith("Ni"):
            species.append(0)
            sgn = 1.0 if lb == "Ni+" else -1.0
            magmoms.append([0, 0, sgn])
            start_mag.append(0.5 * sgn)
        else:
            species.append(1)
            magmoms.append([0, 0, 0])
            start_mag.append(0.0)
    positions = pscaled @ pcell
    return Mineral(
        name="NiO", cell=np.asarray(pcell), positions=positions, species=species,
        symbols=symbols, pseudos=pseudos, magmoms=np.array(magmoms, float),
        start_mag=start_mag, kmesh=kmesh, ecut_ry=ecut_ry, degauss_ry=degauss_ry,
        nspin=2, pseudo_kind="NC", afm=True,
        note="rocksalt a=%.2f, type-II AFM [111]" % a)


def _corundum(name, a_hex, c_hex, z_M, x_O, symbol_M, pseudo_M, spin_pattern,
              ecut_ry, degauss_ry, kmesh, m0=0.5):
    """Corundum-structure AFM (R-3c, #167): 10-atom rhombohedral primitive
    (4 M + 6 O). The AFM order is k=0 (magnetic cell == chemical primitive):
    the four M along the 3-fold (rhombohedral [111] == hex c) get signs from
    ``spin_pattern`` ordered by their projection on that axis."""
    from ase.spacegroup import crystal

    at = crystal(
        symbols=[symbol_M, "O"],
        basis=[(0.0, 0.0, z_M), (x_O, 0.0, 0.25)],
        spacegroup=167, cellpar=[a_hex, a_hex, c_hex, 90, 90, 120],
        primitive_cell=True,
    )
    cell = np.asarray(at.get_cell())
    positions = at.get_positions()
    chem = at.get_chemical_symbols()
    # 3-fold axis of the rhombohedral primitive == sum of the three cell vectors
    axis = cell.sum(axis=0)
    axis = axis / np.linalg.norm(axis)
    m_idx = [i for i, s in enumerate(chem) if s == symbol_M]
    proj = sorted(m_idx, key=lambda i: float(positions[i] @ axis))
    sign_of = {}
    for rank, i in enumerate(proj):
        sign_of[i] = spin_pattern[rank]

    symbols = [symbol_M, "O"]
    pseudos = [pseudo_M, "O_ONCV_PBE-1.2.upf"]
    species, magmoms, start_mag = [], [], []
    for i, s in enumerate(chem):
        if s == symbol_M:
            species.append(0)
            sgn = float(sign_of[i])
            magmoms.append([0, 0, sgn])
            start_mag.append(m0 * sgn)
        else:
            species.append(1)
            magmoms.append([0, 0, 0])
            start_mag.append(0.0)
    return Mineral(
        name=name, cell=cell, positions=positions, species=species,
        symbols=symbols, pseudos=pseudos, magmoms=np.array(magmoms, float),
        start_mag=start_mag, kmesh=kmesh, ecut_ry=ecut_ry, degauss_ry=degauss_ry,
        nspin=2, pseudo_kind="NC", afm=True,
        note="corundum R-3c a=%.3f c=%.3f, AFM %s" % (
            a_hex, c_hex, "".join("+" if x > 0 else "-" for x in spin_pattern)))


def hematite(ecut_ry: float = 70.0, degauss_ry: float = 0.01,
             kmesh=(4, 4, 4)) -> Mineral:
    """alpha-Fe2O3 hematite, corundum, Fe up-up-down-down along c."""
    return _corundum(
        "Fe2O3", a_hex=5.038, c_hex=13.772, z_M=0.3553, x_O=0.3059,
        symbol_M="Fe", pseudo_M="Fe_ONCV_PBE-1.2.upf",
        spin_pattern=[+1, +1, -1, -1], ecut_ry=ecut_ry, degauss_ry=degauss_ry,
        kmesh=kmesh)


def eskolaite(ecut_ry: float = 70.0, degauss_ry: float = 0.01,
              kmesh=(4, 4, 4)) -> Mineral:
    """Cr2O3 eskolaite, corundum, Cr +-+- along c."""
    return _corundum(
        "Cr2O3", a_hex=4.958, c_hex=13.594, z_M=0.3475, x_O=0.3061,
        symbol_M="Cr", pseudo_M="Cr_ONCV_PBE-1.2.upf",
        spin_pattern=[+1, -1, +1, -1], ecut_ry=ecut_ry, degauss_ry=degauss_ry,
        kmesh=kmesh)


def pyrite(ecut_ry: float = 45.0, degauss_ry: float = 0.01,
           kmesh=(4, 4, 4)) -> Mineral:
    """FeS2 pyrite (Pa-3, #205), diamagnetic low-spin -> nspin=1. Primitive
    cubic cell already holds 4 Fe + 8 S = 12 atoms. USPP/PAW path: Fe PAW +
    S ultrasoft (the S pseudo exists only as USPP)."""
    from ase.spacegroup import crystal

    a, x = 5.416, 0.3847
    at = crystal(symbols=["Fe", "S"], basis=[(0.0, 0.0, 0.0), (x, x, x)],
                 spacegroup=205, cellpar=[a, a, a, 90, 90, 90],
                 primitive_cell=True)
    cell = np.asarray(at.get_cell())
    positions = at.get_positions()
    chem = at.get_chemical_symbols()
    species = [0 if s == "Fe" else 1 for s in chem]
    symbols = ["Fe", "S"]
    pseudos = ["Fe.pbe-spn-kjpaw_psl.1.0.0.UPF", "s_pbe_v1.4.uspp.F.UPF"]
    magmoms = np.zeros((len(species), 3))
    start_mag = [0.0] * len(species)
    return Mineral(
        name="FeS2", cell=cell, positions=positions, species=species,
        symbols=symbols, pseudos=pseudos, magmoms=magmoms, start_mag=start_mag,
        kmesh=kmesh, ecut_ry=ecut_ry, degauss_ry=degauss_ry, nspin=1,
        pseudo_kind="USPP", afm=False, rho_factor=10.0,
        note="pyrite Pa-3 a=%.3f, diamagnetic (nspin=1), Fe PAW + S USPP" % a)


REGISTRY = {"nio": nio, "hematite": hematite, "eskolaite": eskolaite,
            "pyrite": pyrite}


def build(name: str) -> Mineral:
    return REGISTRY[name]()


if __name__ == "__main__":
    import spglib
    for key in REGISTRY:
        m = build(key)
        dset = spglib.get_symmetry_dataset(
            (m.cell, m.positions @ np.linalg.inv(m.cell),
             [s for s in m.species]))
        sg = dset.international if hasattr(dset, "international") else dset["international"]
        nmag = int(sum(1 for r in m.magmoms if np.linalg.norm(r) > 0))
        net = float(np.sum([r[2] for r in m.magmoms]))
        print(f"{m.name:8s} nat={m.nat:2d} mag_sites={nmag} net_spin={net:+.1f} "
              f"spg(nonmag)={sg} kmesh={m.kmesh} ecut={m.ecut_ry} kind={m.pseudo_kind}")
        print(f"         note: {m.note}")
        # sanity: nearest neighbour M-O distance
        from scipy.spatial.distance import cdist
        d = cdist(m.positions, m.positions)
        np.fill_diagonal(d, 9e9)
        print(f"         min interatomic dist = {d.min():.3f} A")
