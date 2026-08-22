"""Build the Elk 11.0.2 EFG input dir for anatase TiO2 on asus.

Reuses the exact species files from the validated rutile run (~/tio2_efg/{Ti,O}.in) so the Elk
methodology is identical; only the muffin-tin rmt is patched to match gradwave's spheres
(Ti 1.06 A=2.0032 bohr, O 0.824 A=1.5571 bohr). Same avec + fractional coords as the gradwave run.
"""
import os

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "tio2_efg")          # validated rutile species files
DST = os.path.join(HOME, "efg_multimat", "anatase_elk")
os.makedirs(DST, exist_ok=True)

RMT = {"Ti.in": 2.0032, "O.in": 1.5571}       # bohr, = gradwave R_MT (1.06 / 0.824 A)


def patch_rmt(name):
    with open(os.path.join(SRC, name)) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if "rmt" in ln and "rminsp" in ln:      # the "rminsp, rmt, rmaxsp, nrmt" line
            toks = ln.split()
            toks[1] = f"{RMT[name]:.4f}"
            # rebuild preserving the trailing comment
            comment = ln[ln.index(":"):] if ":" in ln else ""
            lines[i] = "  " + "   ".join(toks[:4]) + "    " + comment
            break
    with open(os.path.join(DST, name), "w") as f:
        f.writelines(lines)


for n in RMT:
    patch_rmt(n)

# anatase primitive cell (Bohr) + fractional coords (identical to anatase_efg.py)
AVEC = [[-3.575551, 3.575551, 8.989993],
        [3.575551, -3.575551, 8.989993],
        [3.575551, 3.575551, -8.989993]]
TI = [(0.0, 0.0, 0.0), (0.75, 0.25, 0.5)]
O = [(0.2081, 0.2081, 0.0), (0.9581, 0.4581, 0.5),
     (0.5419, 0.0419, 0.5), (0.7919, 0.7919, 0.0)]

elk = []
elk += ["tasks", "  0", "  115", ""]
elk += ["lmaxi", "  2", ""]
elk += ["xctype", "  3", ""]              # PW-LDA, matches gradwave
elk += ["scale", "  1.0", ""]
elk += ["avec"]
for v in AVEC:
    elk += [f"  {v[0]:.6f}   {v[1]:.6f}   {v[2]:.6f}"]
elk += [""]
elk += ["atoms", "  2", "  Ti.in", f"  {len(TI)}"]
for p in TI:
    elk += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0"]
elk += ["  O.in", f"  {len(O)}"]
for p in O:
    elk += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}    0.0 0.0 0.0"]
elk += ["", "ngridk", "  4  4  4", "", "rgkmax", "  7.0", ""]

with open(os.path.join(DST, "elk.in"), "w") as f:
    f.write("\n".join(elk) + "\n")
print("wrote", DST)
print(open(os.path.join(DST, "elk.in")).read())
print("--- Ti.in rmt line ---")
for ln in open(os.path.join(DST, "Ti.in")):
    if "rmt" in ln and "rminsp" in ln:
        print(ln.rstrip())
for ln in open(os.path.join(DST, "O.in")):
    if "rmt" in ln and "rminsp" in ln:
        print(ln.rstrip())
