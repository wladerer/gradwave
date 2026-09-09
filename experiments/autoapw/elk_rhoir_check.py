# ruff: noqa: E402, E501
"""Does Elk's interstitial density rhoir also peak to ~+25 e inside the small Ti MT?
Reads rhoir (real-space, e/Bohr^3) from STATE.OUT and integrates it inside each MT sphere.
Usage: uv run python elk_rhoir_check.py <STATE.OUT>  (Ti+2O geometry hardcoded, cell 13 Bohr)"""
import struct
import sys

import numpy as np

LL = 13.0  # Bohr
TI = np.array([0.5, 0.5, 0.5]) * LL
O_S = TI + np.array([1.75 / 0.529177210903, 0.0, 0.0])
O_L = TI + np.array([0.0, 2.30 / 0.529177210903, 0.0])


def records(fh):
    while True:
        head = fh.read(4)
        if len(head) < 4:
            return
        (n,) = struct.unpack("<i", head)
        data = fh.read(n)
        fh.read(4)
        yield data


def main():
    path = sys.argv[1]
    with open(path, "rb") as fh:
        recs = list(records(fh))
    nspecies = struct.unpack("<i", recs[2])[0]
    lmmaxo = struct.unpack("<i", recs[3])[0]
    nrmtmax = struct.unpack("<i", recs[4])[0]
    natoms = [struct.unpack("<i", recs[6 + 5 * s])[0] for s in range(nspecies)]
    natmtot = sum(natoms)
    base2 = 6 + 5 * nspecies
    ngridg = np.frombuffer(recs[base2][:12], "<i4")
    ngtot = int(np.prod(ngridg))
    rho_idx = 17 + 5 * nspecies
    nrf = lmmaxo * nrmtmax * natmtot
    rhoir = np.frombuffer(recs[rho_idx][nrf * 8: nrf * 8 + ngtot * 8], "<f8").reshape(tuple(ngridg), order="F")
    print(f"# ngridg={tuple(int(x) for x in ngridg)} ngtot={ngtot}  rhoir sum*dV = {rhoir.sum()*LL**3/ngtot:.4f} e (interstitial+MT continuation)")
    dV = LL**3 / ngtot
    ax = [np.arange(ngridg[i]) / ngridg[i] * LL for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    for name, c, R in (("Ti", TI, 0.90), ("O_short", O_S, 1.00), ("O_long", O_L, 1.00)):
        d = np.sqrt((((X - c[0] + LL / 2) % LL) - LL / 2)**2
                    + (((Y - c[1] + LL / 2) % LL) - LL / 2)**2
                    + (((Z - c[2] + LL / 2) % LL) - LL / 2)**2)
        q_in = float(rhoir[d < R].sum() * dV)
        print(f"  {name:8s} R={R:.2f} Bohr:  Elk rhoir inside = {q_in:+.4f} e")


if __name__ == "__main__":
    main()
