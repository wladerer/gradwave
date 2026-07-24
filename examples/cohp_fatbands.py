"""COHP fat-band structure for diamond and GaAs.

Produces, for each material, a LOBSTER-style two-panel figure:

  left  — the band structure along a high-symmetry path, every (k, band)
          coloured by its Crystal-Orbital-Hamilton-Population weight on the
          nearest-neighbour bond (bonding < 0 blue, antibonding > 0 red), with
          irrep (Mulliken) labels annotated at the high-symmetry points;
  right — the energy-resolved COHP curve -COHP(E) on the SCF k-mesh, the
          standard bonding/antibonding descriptor, sharing the energy axis.

The band-path COHP is the k/band-decomposed operator-route COHP: for each path
k-point we solve the frozen-potential Hamiltonian keeping the eigenvectors, then
reuse the per-k helpers in gradwave.postscf.cohp to form the AO-basis Hamiltonian
H~ = <phi~|H^|phi~> and the per-eigenstate bond weight
    w_b = 2 Re sum_{p in I, q in J} conj(P_bp) H~_pq P_bq.
This is exactly the array COHP.band_cohp exposes on the SCF mesh, evaluated
instead along the band path so it can decorate the connected bands.

Quantitative note (see the gradwave.postscf.cohp docstring): the ABSOLUTE
per-bond ICOHP is not yet calibrated to LOBSTER (pseudo-atomic basis, Bloch
sublattice pairing), so treat the colour/curve as qualitative — sign and
bonding/antibonding shape are correct, the eV magnitude is not validated.

Run:  uv run python examples/cohp_fatbands.py [--outdir out] [--npoints 80]
Needs matplotlib (the `analysis` optional-dependency group).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from gradwave.constants import HBAR2_2M
from gradwave.core.hamiltonian import HamiltonianK, build_projector_data, projectors
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere
from gradwave.postscf.bands import bands_along_ase_path
from gradwave.postscf.cohp import (
    _htilde_operator,
    _pair_block_weights,
    cohp,
    o_inv_sqrt,
)
from gradwave.postscf.irreps import band_irreps
from gradwave.postscf.pdos import _ao_projectors_k, _atomic_columns
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.davidson import davidson

RY = 13.605693122994
PSEUDOS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"
_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@dataclass
class Material:
    name: str
    a: float                 # conventional lattice constant [A]
    pseudos: list            # upf filenames, one per species index
    species_of_atom: list    # 0-based species index per atom
    symbols: str             # ASE chemical formula for the 2-atom cell
    ecut_ry: float
    nbands: int
    kmesh: tuple
    smearing: str
    width: float
    bond: tuple = (0, 1)     # nearest-neighbour atom pair to resolve
    path: str = "LGXUG"      # FCC / zincblende Brillouin-zone path


# PseudoDojo NC-SR PBE "standard" pseudos: unlike the SG15 ONCV set they retain
# the PP_PSWFC atomic orbitals the COHP projection needs. Ga/As carry 3d
# semicore (13/15 valence electrons), hence the higher cutoff and band count.
MATERIALS = [
    Material("diamond", 3.567, ["PD_C_PBE_std.upf"], [0, 0], "C2",
             ecut_ry=60.0, nbands=8, kmesh=(6, 6, 6),
             smearing="none", width=0.0),
    Material("GaAs", 5.653, ["PD_Ga_PBE_std.upf", "PD_As_PBE_std.upf"],
             [0, 1], "GaAs", ecut_ry=80.0, nbands=20, kmesh=(6, 6, 6),
             smearing="gaussian", width=0.05),
]


def build(mat: Material):
    """Converge the SCF and return (SCFResult, ASE Atoms)."""
    cell = mat.a / 2 * _FCC
    frac = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]])
    upfs = [parse_upf(str(PSEUDOS / p)) for p in mat.pseudos]
    system = setup_system(
        cell, frac @ cell, mat.species_of_atom, upfs, ecut=mat.ecut_ry * RY,
        kmesh=mat.kmesh, nbands=mat.nbands, use_symmetry=True,
    )
    res = scf(system, PBE(), smearing=mat.smearing, width=mat.width,
              etol=1e-8, rhotol=1e-7, verbose=False)
    atoms = Atoms(mat.symbols, scaled_positions=frac, cell=cell, pbc=True)
    return res, atoms


@torch.no_grad()
def cohp_fatbands(res, kpts_frac, bond, nbands):
    """Per-(path-k, band) COHP weight on `bond` and the matching eigenvalues.

    Mirrors the frozen-potential per-k solve of gradwave.postscf.bands, but keeps
    the eigenvectors so the operator-route COHP helpers can weight each state.
    Returns (eigenvalues (nk, nb) [eV], weights (nk, nb) [eV], bonding<0).
    """
    system = res.system
    grid = system.grid
    device = res.v_eff.device
    cols = _atomic_columns(system)
    atom_of = np.array([c.atom for c in cols])
    i, j = bond
    g_spin = 2.0 if getattr(res, "nspin", 1) == 1 else 1.0

    beta_ls = [[b.l for b in u.betas] for u in system.upfs]
    dij = [torch.as_tensor(u.dij, dtype=RDTYPE, device=device) for u in system.upfs]

    kpts = np.asarray(kpts_frac, dtype=float)
    eig = np.empty((len(kpts), nbands))
    wgt = np.empty((len(kpts), nbands))

    for ik, k in enumerate(kpts):
        sph = build_gsphere(grid, system.ecut, k, device=device)
        q_amp = np.sqrt(sph.kpg2.cpu().numpy())
        beta_tables = [torch.as_tensor(beta_form_factors(u, q_amp), dtype=RDTYPE,
                                       device=device) for u in system.upfs]
        pd = build_projector_data(sph, system.species_of_atom, beta_tables,
                                  beta_ls, dij, grid.volume)
        p = projectors(pd, system.positions)
        h = HamiltonianK(sph, grid.shape, res.v_eff, pd, p)
        c0 = torch.zeros(nbands, sph.npw, dtype=CDTYPE, device=device)
        c0[torch.arange(nbands), torch.arange(nbands)] = 1.0
        out = davidson(h.apply, c0, HBAR2_2M * sph.kpg2, tol=1e-9, max_iter=200)
        c = out.eigenvectors                                   # (nb, npw)
        e = out.eigenvalues

        # operator-route AO-basis Hamiltonian and Loewdin amplitudes at this k
        qao = _ao_projectors_k(system, sph, cols, device)      # (nproj, npw)
        overlap = torch.einsum("ig,jg->ij", qao.conj(), qao)
        ois = o_inv_sqrt(overlap)
        becp = torch.einsum("bg,pg->bp", c, qao.conj())
        proj = becp @ ois.conj()                               # (nb, nproj)
        htilde = _htilde_operator(qao, ois, h.apply)
        w = _pair_block_weights(proj, htilde, atom_of, i, j, 2.0) * g_spin

        eig[ik] = e.cpu().numpy()
        wgt[ik] = w
    return eig, wgt


def irreps_at_specials(res, special_points, names, nbands):
    """Irrep clusters keyed by special-point name.

    `special_points` maps name -> fractional k (from the SAME bandpath used for
    the plot); `names` are the special points to label. Keying by name (not by an
    x taken from a differently-sampled path) keeps the annotations aligned with
    the band-axis ticks.
    """
    out = {}
    for name in names:
        if name in out or name not in special_points:
            continue
        try:
            ki = band_irreps(res, special_points[name], nbands=nbands)
        except Exception as exc:                               # noqa: BLE001
            print(f"    irreps at {name} failed: {exc}")
            continue
        out[name] = [{"e": float(np.mean(c.energies)), "label": c.label,
                      "dim": c.dim, "warning": c.warning} for c in ki.clusters]
    return out


def run_material(mat: Material, outdir: Path, npoints: int) -> dict:
    print(f"[{mat.name}] SCF ({mat.kmesh} mesh, {mat.ecut_ry:.0f} Ry) ...",
          flush=True)
    res, atoms = build(mat)
    print(f"[{mat.name}] SCF converged={res.converged} "
          f"in {res.n_iter} iters, E_F={res.fermi:.3f} eV", flush=True)

    # band path: reuse the tested reference/x/labels, add COHP weights ourselves
    bs = bands_along_ase_path(res, atoms, path=mat.path, npoints=npoints,
                              nbands=mat.nbands)
    bp = atoms.cell.bandpath(path=mat.path, npoints=npoints)
    print(f"[{mat.name}] COHP fat-bands along {mat.path} "
          f"({len(bp.kpts)} k) ...", flush=True)
    eig, wgt = cohp_fatbands(res, bp.kpts, mat.bond, mat.nbands)

    print(f"[{mat.name}] irreps at special points ...", flush=True)
    tick_names = list(dict.fromkeys(lab for _, lab in bs.labels))
    irr = irreps_at_specials(res, bp.special_points, tick_names, mat.nbands)
    for name, clusters in irr.items():
        labs = ", ".join(c["label"] for c in clusters[:6])
        print(f"    {name}: {labs} ...", flush=True)

    print(f"[{mat.name}] energy-resolved COHP on SCF mesh ...", flush=True)
    ch = cohp(res, pairs=[mat.bond], rcut=3.0, width=0.1, npoints=800)
    blab = f"{mat.bond[0] + 1}-{mat.bond[1] + 1}"

    data = {
        "material": mat.name,
        "reference_eV": float(bs.reference),
        "fermi_eV": float(res.fermi),
        "x": np.asarray(bs.x).tolist(),
        "labels": [[float(xt), lab] for xt, lab in bs.labels],
        "eigenvalues_eV": eig.tolist(),
        "cohp_weight": wgt.tolist(),
        "bond": list(mat.bond),
        "irreps": irr,
        "curve_energy_eV": ch.energy_eV.tolist(),
        "curve_cohp": ch.pair_cohp[blab].tolist(),
        "pair_icohp_eV": float(ch.pair_icohp[blab]),
        "spilling": float(ch.spilling),
        "charge_spilling": float(ch.charge_spilling),
    }
    jpath = outdir / f"{mat.name}_cohp_fatbands.json"
    jpath.write_text(json.dumps(data))
    print(f"[{mat.name}] wrote {jpath}", flush=True)
    return data


def plot_material(data: dict, outdir: Path, window=(-18, 12)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    name = data["material"]
    title = data.get("title", name)
    ref = data["reference_eV"]
    x = np.asarray(data["x"])
    eig = np.asarray(data["eigenvalues_eV"]) - ref
    wgt = np.asarray(data["cohp_weight"])
    nb = eig.shape[1]

    # symmetric diverging colour scale from a robust weight magnitude
    wmax = float(np.percentile(np.abs(wgt), 98)) or 1.0
    norm = TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)
    cmap = plt.get_cmap("coolwarm")

    fig, (axb, axc) = plt.subplots(
        1, 2, figsize=(8.4, 5.6), sharey=True, layout="constrained",
        gridspec_kw={"width_ratios": [3, 1.2]})

    # left: COHP fat bands
    for b in range(nb):
        pts = np.column_stack([x, eig[:, b]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cseg = 0.5 * (wgt[:-1, b] + wgt[1:, b])
        lc = LineCollection(segs, cmap=cmap, norm=norm, zorder=2)
        lc.set_array(cseg)
        lc.set_linewidth(2.0)
        axb.add_collection(lc)

    # merge special points that share an x (FCC zone-boundary breaks, e.g. X|U)
    xmax = float(x[-1])
    merged = {}  # x -> [names in path order]
    for xt, lab in data["labels"]:
        merged.setdefault(round(xt, 6), []).append(lab)
    for xt in merged:
        axb.axvline(xt, color="0.8", lw=0.6, zorder=0)
    axb.axhline(0.0, color="0.5", lw=0.7, ls="--", zorder=1)

    # irrep annotations: one label set per tick, placed at the band-axis x, using
    # the first named special point at that x that has irreps
    irr = data["irreps"]
    for xt, names in merged.items():
        pname = next((n for n in names if n in irr), None)
        if pname is None:
            continue
        at_end = xt > 0.98 * xmax
        dx = -0.01 * xmax if at_end else 0.01 * xmax
        ha = "right" if at_end else "left"
        placed = []
        for cl in sorted(irr[pname], key=lambda c: c["e"]):
            y = cl["e"] - ref
            if not (window[0] < y < window[1]) or cl["label"] in ("?", ""):
                continue
            if placed and min(abs(y - p) for p in placed) < 0.5:
                continue  # skip near-overlapping labels for legibility
            placed.append(y)
            axb.annotate(cl["label"], (xt + dx, y), fontsize=6.5, ha=ha,
                         va="center", color="#20304a", zorder=4,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                   ec="0.7", lw=0.4, alpha=0.8))

    axb.set_xticks(list(merged))
    axb.set_xticklabels(["|".join(names).replace("G", "Γ")
                         for names in merged.values()])
    axb.set_xlim(float(x[0]), xmax)
    axb.set_ylim(*window)
    axb.set_ylabel("E − E$_{ref}$ (eV)")
    bi, bj = data["bond"]
    axb.set_title(f"{title}: COHP fat bands (bond {bi + 1}–{bj + 1})",
                  fontsize=11)

    # right: -COHP(E) curve (bonding to the right), shared energy axis
    ce = np.asarray(data["curve_energy_eV"]) - ref
    cc = -np.asarray(data["curve_cohp"])       # plot -COHP: bonding positive
    axc.plot(cc, ce, color="0.15", lw=1.1, zorder=3)
    axc.axvline(0.0, color="0.6", lw=0.6, zorder=1)
    axc.axhline(0.0, color="0.5", lw=0.7, ls="--", zorder=1)
    axc.fill_betweenx(ce, cc, 0, where=cc > 0, color="#3b6fb0", alpha=0.35,
                      label="bonding")
    axc.fill_betweenx(ce, cc, 0, where=cc < 0, color="#c0392b", alpha=0.35,
                      label="antibonding")
    axc.set_xlabel("−COHP (arb.)")
    axc.set_title(f"ICOHP = {data['pair_icohp_eV']:.2f} eV", fontsize=9)
    axc.legend(fontsize=6.5, loc="lower right", frameon=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axc, fraction=0.09, pad=0.16)
    cb.set_label("COHP weight  (← bonding | antibonding →)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{title} — k/band-decomposed COHP  "
        f"(spilling {data['spilling']:.2f}, charge {data['charge_spilling']:.2f}; "
        "magnitude qualitative)", fontsize=8)
    ppath = outdir / f"{name}_cohp_fatbands.png"
    fig.savefig(ppath, dpi=180)
    plt.close(fig)
    print(f"[{name}] wrote {ppath}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--npoints", type=int, default=80,
                    help="k-points along the band path")
    ap.add_argument("--only", default=None, help="run one material by name")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mats = [m for m in MATERIALS if args.only is None or m.name == args.only]
    for mat in mats:
        data = run_material(mat, outdir, args.npoints)
        plot_material(data, outdir)


if __name__ == "__main__":
    main()
