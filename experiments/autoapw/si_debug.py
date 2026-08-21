"""Throwaway diagnostic for the shift-invert solver on the synthetic unit-test pencil."""
import numpy as np

from gradwave.flapw.lapw import solve_geneig, solve_geneig_shift_invert


def pencil(dim, eigs, rng):
    b = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    s = b @ b.conj().T + dim * np.eye(dim)
    lc = np.linalg.cholesky(s)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(m)
    v = np.linalg.solve(lc.conj().T, q)
    h = s @ v @ np.diag(eigs) @ v.conj().T @ s
    return 0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T)


rng = np.random.default_rng(3)
dim, nb = 90, 14
eigs = np.sort(rng.uniform(-5.0, 20.0, dim))
h, s = pencil(dim, eigs, rng)
ref = solve_geneig(h, s, nb)
print("ref[:5]=", ref[:5], "ref[-1]=", ref[-1], flush=True)
out = solve_geneig_shift_invert(h, s, nb, sigma=float(ref[0]) - 1.0)
print("out is None?", out is None, flush=True)
if out is not None:
    print("max|dEv|=", np.abs(np.sort(out[0]) - ref).max(), flush=True)
