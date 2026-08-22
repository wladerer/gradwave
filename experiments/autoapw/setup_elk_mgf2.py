"""Build the Elk 11.0.2 EFG input dir for MgF2 (rutile) on asus.

Stock Elk Mg.in / F.in species; rmt forced to gradwave's spheres (Mg 1.0 A = 1.8897 bohr,
F 0.80 A = 1.5118 bohr). Same avec + fractional coords as mgf2_efg.py.
"""
import os

HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "efg_multimat", "mgf2_elk")
os.makedirs(DST, exist_ok=True)
SPDIR = os.path.join(HOME, "github", "elk-11.0.2", "species")
RMT = {"Mg.in": 1.8897, "F.in": 1.5118}


def patch_rmt(name):
    with open(os.path.join(SPDIR, name)) as f:
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

A, C, U = 8.73242, 5.69941, 0.303
MG = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
F = [(U, U, 0.0), (1 - U, 1 - U, 0.0), (0.5 + U, 0.5 - U, 0.5), (0.5 - U, 0.5 + U, 0.5)]
elk = ["tasks", "  0", "  115", "", "lmaxi", "  2", "", "xctype", "  3", "",
       "scale", "  1.0", "", "avec",
       f"  {A:.6f}   0.0   0.0", f"  0.0   {A:.6f}   0.0", f"  0.0   0.0   {C:.6f}",
       "", "atoms", "  2", "  Mg.in", f"  {len(MG)}"]
for p in MG:
    elk += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0"]
elk += ["  F.in", f"  {len(F)}"]
for p in F:
    elk += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0"]
elk += ["", "ngridk", "  4  4  6", "", "rgkmax", "  7.0", ""]

with open(os.path.join(DST, "elk.in"), "w") as f:
    f.write("\n".join(elk) + "\n")
print("wrote", DST)
for nm in ("Mg.in", "F.in"):
    for ln in open(os.path.join(DST, nm)):
        if "rmt" in ln and "rminsp" in ln:
            print(nm, ln.rstrip())
