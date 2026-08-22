"""Build Elk 11.0.2 EFG input dirs (tasks 0,115) for the main-group NMR set on asus.

One dir per material under ~/efg_mgroup/<name>_elk. Stock Elk species (semicore in valence = full
all-electron antishielding reference), rmt FORCED to gradwave's spheres, rgkmax MATCHED to
gradwave's effective R_min·Gmax(ecut) so both codes use the same basis cutoff. LDA (xctype 3),
lmaxi 2, ngridk 4 4 4. NO src change. Run: python setup_elk_mgroup.py, then run elk in each dir.
"""
import math
import os

BOHR = 0.5291772109
HART = 27.211386
ELK = os.path.expanduser("~/github/elk-11.0.2")
BASE = os.path.expanduser("~/efg_mgroup")


def hexavec(a, c):
    a, c = a / BOHR, c / BOHR
    return [[a, 0.0, 0.0], [-0.5 * a, math.sqrt(3) / 2 * a, 0.0], [0.0, 0.0, c]]


def orthoavec(a, b, c):
    return [[a / BOHR, 0.0, 0.0], [0.0, b / BOHR, 0.0], [0.0, 0.0, c / BOHR]]


# name -> (avec, {species: (rmt_ang, [fracs])}, ecut_eV, order)
MATS = {
    "hbn": dict(
        avec=hexavec(2.504, 6.661), ecut=400.0,
        species={"B": (0.70, [(1 / 3, 2 / 3, 1 / 4), (2 / 3, 1 / 3, 3 / 4)]),
                 "N": (0.70, [(2 / 3, 1 / 3, 1 / 4), (1 / 3, 2 / 3, 3 / 4)])}),
    "aln": dict(
        avec=hexavec(3.111, 4.978), ecut=300.0,
        species={"Al": (0.95, [(1 / 3, 2 / 3, 0.0), (2 / 3, 1 / 3, 0.5)]),
                 "N": (0.85, [(1 / 3, 2 / 3, 0.3821), (2 / 3, 1 / 3, 0.8821)])}),
    "li3n": dict(
        avec=hexavec(3.648, 3.875), ecut=250.0,
        species={"Li": (0.90, [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.0), (2 / 3, 1 / 3, 0.0)]),
                 "N": (0.90, [(0.0, 0.0, 0.0)])}),
    "nano2": dict(
        avec=orthoavec(3.5653, 5.5728, 5.3846), ecut=350.0,
        species={"Na": (1.05, [(0.0, 0.5881, 0.0), (0.5, 0.0881, 0.5)]),
                 "N": (0.62, [(0.0, 0.1224, 0.0), (0.5, 0.6224, 0.5)]),
                 "O": (0.62, [(0.0, 0.0, 0.1962), (0.5, 0.5, 0.6962),
                              (0.0, 0.0, 0.8038), (0.5, 0.5, 0.3038)])}),
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


def build(name, m):
    dst = os.path.join(BASE, f"{name}_elk")
    os.makedirs(dst, exist_ok=True)
    gmax = math.sqrt(2 * m["ecut"] / HART)                          # Bohr^-1
    rmin_bohr = min(r / BOHR for r, _ in m["species"].values())
    rgkmax = rmin_bohr * gmax
    for sp, (rmt_ang, _) in m["species"].items():
        patch_rmt(os.path.join(ELK, "species", f"{sp}.in"),
                  os.path.join(dst, f"{sp}.in"), rmt_ang / BOHR)
    elk = ["tasks", "  0", "  115", "", "lmaxi", "  2", "", "xctype", "  3", "",
           "scale", "  1.0", "", "avec"]
    for v in m["avec"]:
        elk.append(f"  {v[0]:.6f}   {v[1]:.6f}   {v[2]:.6f}")
    elk += ["", "atoms", f"  {len(m['species'])}"]
    for sp, (_, fracs) in m["species"].items():
        elk += [f"  {sp}.in", f"  {len(fracs)}"]
        for p in fracs:
            elk.append(f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0")
    elk += ["", "ngridk", "  4  4  4", "", "rgkmax", f"  {rgkmax:.4f}", ""]
    with open(os.path.join(dst, "elk.in"), "w") as f:
        f.write("\n".join(elk) + "\n")
    print(f"{name}: {dst} rgkmax={rgkmax:.3f} (R_min={rmin_bohr:.3f}b, ecut={m['ecut']})")


if __name__ == "__main__":
    for name, m in MATS.items():
        build(name, m)
