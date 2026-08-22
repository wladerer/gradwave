"""Build the Elk 11.0.2 EFG input dir for alpha-Al2O3 corundum on asus.

Al.in = stock Elk species (2p IN VALENCE => full all-electron antishielding reference).
O.in  = the validated rutile O.in (same methodology as the TiO2 references).
rmt patched to gradwave's spheres: Al 0.97 A = 1.8331 bohr, O 0.824 A = 1.5571 bohr.
Same avec + fractional coords as corundum_efg.py.
"""
import os

HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "efg_multimat", "corundum_elk")
os.makedirs(DST, exist_ok=True)
SRC = {"Al.in": os.path.join(HOME, "github", "elk-11.0.2", "species", "Al.in"),
       "O.in": os.path.join(HOME, "tio2_efg", "O.in")}
RMT = {"Al.in": 1.8331, "O.in": 1.5571}


def patch_rmt(name):
    with open(SRC[name]) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if "rmt" in ln and "rminsp" in ln:
            toks = ln.split()
            toks[1] = f"{RMT[name]:.4f}"
            comment = ln[ln.index(":"):] if ":" in ln else ""
            lines[i] = "  " + "   ".join(toks[:4]) + "    " + comment
            break
    with open(os.path.join(DST, name), "w") as f:
        f.writelines(lines)


for n in RMT:
    patch_rmt(n)

AVEC = [[4.497737, 2.596770, 8.184592],
        [-4.497737, 2.596770, 8.184592],
        [-0.000000, -5.193539, 8.184592]]
AL = [(0.352160, 0.352160, 0.352160), (0.147840, 0.147840, 0.147840),
      (0.647840, 0.647840, 0.647840), (0.852160, 0.852160, 0.852160)]
O = [(0.250000, 0.556240, 0.943760), (0.056240, 0.750000, 0.443760),
     (0.556240, 0.943760, 0.250000), (0.443760, 0.056240, 0.750000),
     (0.943760, 0.250000, 0.556240), (0.750000, 0.443760, 0.056240)]

elk = ["tasks", "  0", "  115", "", "lmaxi", "  2", "", "xctype", "  3", "",
       "scale", "  1.0", "", "avec"]
for v in AVEC:
    elk += [f"  {v[0]:.6f}   {v[1]:.6f}   {v[2]:.6f}"]
elk += ["", "atoms", "  2", "  Al.in", f"  {len(AL)}"]
for p in AL:
    elk += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0"]
elk += ["  O.in", f"  {len(O)}"]
for p in O:
    elk += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0"]
elk += ["", "ngridk", "  4  4  4", "", "rgkmax", "  7.0", ""]

with open(os.path.join(DST, "elk.in"), "w") as f:
    f.write("\n".join(elk) + "\n")
print("wrote", DST)
for ln in open(os.path.join(DST, "Al.in")):
    if "rmt" in ln and "rminsp" in ln:
        print("Al", ln.rstrip())
for ln in open(os.path.join(DST, "O.in")):
    if "rmt" in ln and "rminsp" in ln:
        print("O ", ln.rstrip())
