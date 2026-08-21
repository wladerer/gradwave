"""Little-group reduction factors for the NMR shielding q-directions.

The shielding driver (sigma_shielding / sigma_shielding_dq) uses q along each
sampled reciprocal axis b_i (mesh_n[i] > 1). We compute:
  - full point group |G|
  - little co-group of q along each reciprocal axis b_i  (q_frac = eps * e_i)
  - the per-axis reduction factor = |little co-group of q_i|
  - the k-mesh IBZ counts (full vs little-group-reduced) at a few meshes
"""
import numpy as np
from gradwave.symmetry import (
    find_spacegroup, little_cogroup, star_of_q, little_group_ibz, reduce_mesh,
)

_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def si_fcc(a=5.43):
    cell = a / 2 * _FCC
    pos_cart = np.array([[0.0, 0, 0], [a / 4] * 3])
    frac = pos_cart @ np.linalg.inv(cell)  # -> (0,0,0), (1/4,1/4,1/4)
    return cell, frac, [14, 14]


def tio2():
    A_BOHR = np.array([8.68083, 8.68083, 5.59096])
    BOHR_ANG = 0.52917721067
    cell = np.diag(A_BOHR * BOHR_ANG)  # tetragonal
    U = 0.3048
    frac = np.array([
        (0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
        (U, U, 0.0), (1 - U, 1 - U, 0.0),
        (0.5 + U, 0.5 - U, 0.5), (0.5 - U, 0.5 + U, 0.5)])
    znum = [22, 22, 8, 8, 8, 8]
    return cell, frac, znum


def analyze(name, cell, frac, znum, meshes):
    sg = find_spacegroup(cell, frac, znum)
    print(f"\n========== {name} ==========")
    print(f"  international: {sg.international}")
    print(f"  |point group| = {sg.n_ops} rotations")
    eps = 1e-3  # infinitesimal q along an axis (direction only matters)
    for i in range(3):
        q = np.zeros(3); q[i] = eps
        lg, g0 = little_cogroup(q, sg)
        star, reps = star_of_q(q, sg)
        print(f"  q along b{i+1}=(e{i+1}): |little co-group| = {lg.n_ops:2d}   "
              f"reduction={sg.n_ops/lg.n_ops:.1f}x   |star|={len(star)}  "
              f"(orbit-stab: {len(star)}*{lg.n_ops}={len(star)*lg.n_ops}=={sg.n_ops})")
    # k-mesh IBZ counts
    for mesh in meshes:
        full = int(np.prod(mesh))
        # full-mesh (what _guard currently forces): no reduction
        # little-group IBZ per axis-direction q
        line = f"  mesh {mesh}: full={full}"
        for i in range(3):
            if mesh[i] <= 1:
                continue
            q = np.zeros(3); q[i] = 1.0 / mesh[i]
            kf, w = little_group_ibz(mesh, q, sg, time_reversal=False)
            line += f" | q//b{i+1}: IBZ={len(kf)} ({full/len(kf):.1f}x)"
        print(line)
    # for reference: full-symmetry (Gamma point) IBZ
    for mesh in meshes:
        full = int(np.prod(mesh))
        kf, w = reduce_mesh(mesh, (0, 0, 0), sg, time_reversal=False)
        kf_tr, _ = reduce_mesh(mesh, (0, 0, 0), sg, time_reversal=True)
        print(f"  mesh {mesh}: full-PG IBZ (q=0) = {len(kf)} (noTR), {len(kf_tr)} (TR)")


if __name__ == "__main__":
    c, p, z = si_fcc()
    analyze("Si (diamond, Oh)", c, p, z, [(2, 2, 2), (4, 4, 4), (6, 6, 6)])
    c, p, z = tio2()
    analyze("rutile TiO2 (D4h)", c, p, z, [(2, 2, 2), (4, 4, 4), (6, 6, 6)])
