"""Build Elk 11.0.2 EFG input dirs at an arbitrary ISOTROPIC k-mesh for the converged-k re-read.

MAT selects the material, NGRIDK the isotropic n (n n n) matched to gradwave's kmesh=(n,n,n).
rmt FORCED to gradwave's spheres, rgkmax MATCHED to gradwave's R_min.Gmax(ecut) (same recipe as
setup_elk_mgroup.py), LDA (xctype 3), lmaxi 2, tasks 0 then 115. Dir: ~/efg_kconv/<mat>_k<n>_elk.

Oxide species reuse the validated ~/tio2_efg/{Ti,O}.in (identical methodology to the TiO2 refs);
Al/Mg/F/B/N/Li are stock Elk species (semicore in valence = full all-electron reference).
"""
import math
import os
import sys

BOHR = 0.5291772109
HART = 27.211386
HOME = os.path.expanduser("~")
ELK_SP = os.path.join(HOME, "github", "elk-11.0.2", "species")
TIO2 = os.path.join(HOME, "tio2_efg")
BASE = os.path.join(HOME, "efg_kconv")

U_RUT, U_MGF = 0.3048, 0.303


def hexavec(a, c):
    a, c = a / BOHR, c / BOHR
    return [[a, 0.0, 0.0], [-0.5 * a, math.sqrt(3) / 2 * a, 0.0], [0.0, 0.0, c]]


# name -> avec(Bohr), species {sym: (src_in_path, rmt_ang, [fracs])}, ecut_eV, order-list
MATS = {
    "rutile": dict(
        avec=[[8.68083, 0, 0], [0, 8.68083, 0], [0, 0, 5.59096]], ecut=300.0,
        species=[("Ti", os.path.join(TIO2, "Ti.in"), 1.098,
                  [(0, 0, 0), (0.5, 0.5, 0.5)]),
                 ("O", os.path.join(TIO2, "O.in"), 0.824,
                  [(U_RUT, U_RUT, 0), (1 - U_RUT, 1 - U_RUT, 0),
                   (0.5 + U_RUT, 0.5 - U_RUT, 0.5), (0.5 - U_RUT, 0.5 + U_RUT, 0.5)])]),
    "anatase": dict(
        avec=[[-3.575551, 3.575551, 8.989993], [3.575551, -3.575551, 8.989993],
              [3.575551, 3.575551, -8.989993]], ecut=300.0,
        species=[("Ti", os.path.join(TIO2, "Ti.in"), 1.06,
                  [(0, 0, 0), (0.75, 0.25, 0.5)]),
                 ("O", os.path.join(TIO2, "O.in"), 0.824,
                  [(0.2081, 0.2081, 0), (0.9581, 0.4581, 0.5),
                   (0.5419, 0.0419, 0.5), (0.7919, 0.7919, 0)])]),
    "corundum": dict(
        avec=[[4.497737, 2.596770, 8.184592], [-4.497737, 2.596770, 8.184592],
              [-0.000000, -5.193539, 8.184592]], ecut=300.0,
        species=[("Al", os.path.join(ELK_SP, "Al.in"), 0.97,
                  [(0.352160, 0.352160, 0.352160), (0.147840, 0.147840, 0.147840),
                   (0.647840, 0.647840, 0.647840), (0.852160, 0.852160, 0.852160)]),
                 ("O", os.path.join(TIO2, "O.in"), 0.824,
                  [(0.250000, 0.556240, 0.943760), (0.056240, 0.750000, 0.443760),
                   (0.556240, 0.943760, 0.250000), (0.443760, 0.056240, 0.750000),
                   (0.943760, 0.250000, 0.556240), (0.750000, 0.443760, 0.056240)])]),
    "mgf2": dict(
        avec=[[8.73242, 0, 0], [0, 8.73242, 0], [0, 0, 5.69941]], ecut=300.0,
        species=[("Mg", os.path.join(ELK_SP, "Mg.in"), 1.0,
                  [(0, 0, 0), (0.5, 0.5, 0.5)]),
                 ("F", os.path.join(ELK_SP, "F.in"), 0.80,
                  [(U_MGF, U_MGF, 0), (1 - U_MGF, 1 - U_MGF, 0),
                   (0.5 + U_MGF, 0.5 - U_MGF, 0.5), (0.5 - U_MGF, 0.5 + U_MGF, 0.5)])]),
    "hbn": dict(
        avec=hexavec(2.504, 6.661), ecut=400.0,
        species=[("B", os.path.join(ELK_SP, "B.in"), 0.70,
                  [(1 / 3, 2 / 3, 1 / 4), (2 / 3, 1 / 3, 3 / 4)]),
                 ("N", os.path.join(ELK_SP, "N.in"), 0.70,
                  [(2 / 3, 1 / 3, 1 / 4), (1 / 3, 2 / 3, 3 / 4)])]),
    "li3n": dict(
        avec=hexavec(3.648, 3.875), ecut=250.0,
        species=[("Li", os.path.join(ELK_SP, "Li.in"), 0.90,
                  [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.0), (2 / 3, 1 / 3, 0.0)]),
                 ("N", os.path.join(ELK_SP, "N.in"), 0.90, [(0.0, 0.0, 0.0)])]),
}


def patch_rmt(src, dst, rmt_bohr):
    with open(src) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if "rmt" in ln and "rminsp" in ln:
            toks = ln.split()
            toks[1] = f"{rmt_bohr:.4f}"
            comment = ln[ln.index(":"):] if ":" in ln else ""
            lines[i] = "  " + "   ".join(toks[:4]) + "    " + comment
            break
    with open(dst, "w") as f:
        f.writelines(lines)


def build(name, n):
    m = MATS[name]
    dst = os.path.join(BASE, f"{name}_k{n}_elk")
    os.makedirs(dst, exist_ok=True)
    gmax = math.sqrt(2 * m["ecut"] / HART)
    rmin_bohr = min(rmt / BOHR for _, _, rmt, _ in m["species"])
    rgkmax = rmin_bohr * gmax
    for sym, src, rmt, _ in m["species"]:
        patch_rmt(src, os.path.join(dst, f"{sym}.in"), rmt / BOHR)
    elk = ["tasks", "  0", "  115", "", "lmaxi", "  2", "", "xctype", "  3", "",
           "scale", "  1.0", "", "avec"]
    for v in m["avec"]:
        elk.append(f"  {v[0]:.6f}   {v[1]:.6f}   {v[2]:.6f}")
    elk += ["", "atoms", f"  {len(m['species'])}"]
    for sym, _, _, fracs in m["species"]:
        elk += [f"  {sym}.in", f"  {len(fracs)}"]
        for p in fracs:
            elk.append(f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0")
    elk += ["", "ngridk", f"  {n}  {n}  {n}", "", "rgkmax", f"  {rgkmax:.4f}", ""]
    with open(os.path.join(dst, "elk.in"), "w") as f:
        f.write("\n".join(elk) + "\n")
    print(f"{name} k{n}: {dst} rgkmax={rgkmax:.3f} (R_min={rmin_bohr:.3f}b ecut={m['ecut']})")
    return dst


if __name__ == "__main__":
    mat = os.environ.get("MAT", sys.argv[1] if len(sys.argv) > 1 else "")
    n = int(os.environ.get("NGRIDK", sys.argv[2] if len(sys.argv) > 2 else "6"))
    if mat:
        build(mat, n)
    else:
        for name in MATS:
            build(name, n)
