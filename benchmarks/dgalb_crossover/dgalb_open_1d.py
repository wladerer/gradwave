"""DG-ALB semi-infinite spike (S1 core): open boundary via a BLOCK lead
self-energy, fed by the validated DG-ALB block-tridiagonal Hamiltonian.

PR #265's `experiments/dtn_1d/greens.py` implements the energy-exact open boundary
for a SCALAR 1D lead: surface Green's function g by the quadratic root, self-energy
Sigma = t^2 g, density by a contour integral (validated to match the eigensolver to
5.7e-12). DG-ALB needs the BLOCK generalization: the lead layer is a DG element
(M x M diagonal block H_00) coupled to its neighbour by the SIPG face block H_01, so
the scalar quadratic root becomes a Sancho-Rubio decimation for the block surface GF.
Everything else (the contour density) reuses #265's structure with block matrices.

Validation ladder:
  (1) UNIT TEST: 1x1-block Sancho-Rubio Sigma must reproduce #265's analytic scalar
      lead_sigma(E) to ~1e-12 -> the decimation is correct.
  (2) BULK CONSISTENCY: bands from the DG blocks (H_00 + H_01 e^{ik} + h.c.) must
      reproduce the periodic DG spectrum (folded) -> H_00/H_01 are physical.
  (3) OPEN BOUNDARY: surface LDOS(E) = -Im tr g_surf / pi is positive and lives
      inside the bands (no spurious in-gap states from the open BC).
  (4) BOX-INDEPENDENCE: for a device of k DG elements between two leads, the
      boundary-layer LDOS is independent of k (the leads absorb the rest).

Reuses the DG-ALB block construction from dgalb_spike_1d.py. numpy + torch only.

Run:  uv run python benchmarks/dgalb_crossover/dgalb_open_1d.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgalb_spike_1d as d1  # noqa: E402

torch.set_default_dtype(torch.float64)
CD = torch.complex128
HB = d1.HB


# ---------------------------------------------------------------- scalar reference
def scalar_lead_sigma(E, t, eps0, eta=1e-7):
    """Analytic retarded surface self-energy of a semi-infinite 1D lead (onsite
    eps0, hopping -t): root of t^2 g^2 - (E-eps0) g + 1 = 0. Select the RETARDED
    branch by Im(g) <= 0 with E -> E + i*eta (robust for any |t|, unlike the
    |t^2 g| <= 1 test which fails when |t| > 1)."""
    d = (E + 1j * eta) - eps0
    root = np.sqrt(d * d - 4.0 * t * t + 0j)
    g1 = (d - root) / (2.0 * t * t)
    g2 = (d + root) / (2.0 * t * t)
    g = np.where(g1.imag <= 0.0, g1, g2)
    return t * t * g


# ---------------------------------------------------------------- block surface GF
def sancho_rubio(H00, H01, E, eta=1e-7, tol=1e-12, maxiter=80):
    """Block surface Green's function of a semi-infinite lead of layers H_00 coupled
    by H_01 (layer n -> n+1). Fast decimation (Sancho et al. 1985). Returns the
    LEFT-surface GF (surface of a lead extending to the RIGHT)."""
    n = H00.shape[0]
    z = (E + 1j * eta) * np.eye(n, dtype=complex)
    eps_s = H00.astype(complex).copy()          # surface onsite
    eps = H00.astype(complex).copy()            # bulk onsite
    alpha = H01.astype(complex).copy()          # couples n -> n+1
    beta = H01.conj().T.astype(complex).copy()  # couples n -> n-1
    for _ in range(maxiter):
        g = np.linalg.inv(z - eps)
        agb = alpha @ g @ beta
        bga = beta @ g @ alpha
        eps_s = eps_s + agb
        eps = eps + agb + bga
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if np.abs(alpha).max() < tol:
            break
    return np.linalg.inv(z - eps_s)


def dg_blocks(n_at=6, spacing=2.0, v0=1.0, width=0.4, n_elem=6, buf_frac=1.0,
              nloc=96, ncore=64, M=16, sigma=None):
    """Homogeneous 1D DG-ALB chain -> bulk lead blocks (H_00, H_01) and the periodic
    reference spectrum. All elements identical, so any diagonal block is H_00 and any
    nearest-neighbour block is H_01."""
    L = n_at * spacing
    Vfunc = d1.make_Vfunc(L, n_at, v0, width)
    h = L / n_elem
    if sigma is None:
        sigma = 2.0 * M ** 2
    # homogeneous chain: build ONE element and replicate it, so n_keep is uniform
    # and the assembled H is exactly block-circulant (clean H_00/H_01 extraction).
    elem0 = d1.build_alb(0.0, h, buf_frac * h, Vfunc, nloc, ncore, M)
    elems = [elem0 for _ in range(n_elem)]
    H, herm = d1.assemble(elems, h, sigma)
    n = elem0["n"]
    Hnp = H.numpy()
    H00 = Hnp[:n, :n]
    H01 = Hnp[:n, n:2 * n]                        # coupling element 0 -> 1
    ref = d1.reference_eigs(L, 256, Vfunc)
    dg_spec = np.sort(np.linalg.eigvalsh(Hnp).real)   # true periodic DG spectrum
    return H00, H01, n, herm, ref, sigma, dg_spec


def bulk_bands(H00, H01, nk=24):
    """Band eigenvalues over the 1D BZ from the bulk blocks."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    bands = []
    for k in ks:
        Hk = H00 + H01 * np.exp(1j * k) + H01.conj().T * np.exp(-1j * k)
        bands.append(np.linalg.eigvalsh(Hk).real)
    return np.sort(np.concatenate(bands))


def device_ldos(H00, H01, kdev, E, eta=1e-6):
    """Boundary-layer LDOS of a kdev-element device between two identical leads."""
    n = H00.shape[0]
    D = kdev * n
    Hd = np.zeros((D, D), dtype=complex)
    for i in range(kdev):
        Hd[i * n:(i + 1) * n, i * n:(i + 1) * n] = H00
        if i + 1 < kdev:
            Hd[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = H01
            Hd[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = H01.conj().T
    gL = sancho_rubio(H00, H01, E)               # lead to the left
    gR = sancho_rubio(H00, H01.conj().T, E)      # lead to the right (swap coupling)
    sigL = H01.conj().T @ gL @ H01
    sigR = H01 @ gR @ H01.conj().T
    A = (E + 1j * eta) * np.eye(D, dtype=complex) - Hd
    A[:n, :n] -= sigL
    A[-n:, -n:] -= sigR
    G = np.linalg.inv(A)
    return -np.trace(G[:n, :n]).imag / np.pi


def main():
    print("# DG-ALB semi-infinite spike (open boundary via block lead self-energy)")
    print(f"# reuses dgalb_spike_1d; scalar unit test vs #265 greens.py lead_sigma "
          f"(HB={HB})\n")

    # (1) UNIT TEST: 1x1 block Sancho-Rubio == analytic scalar lead_sigma -----
    t, eps0 = 1.3, -0.4
    H00s = np.array([[eps0 + 2 * t]])            # onsite = eps0 + 2t (band centre)
    H01s = np.array([[-t]])
    print("== (1) unit test: block Sancho-Rubio vs analytic scalar lead_sigma ==")
    print(f"{'E':>7} {'Sigma_block':>26} {'Sigma_analytic':>26} {'|err|':>10}")
    maxerr = 0.0
    for E in (-2.5, -1.0, 0.0, 1.0, 2.5):
        gs = sancho_rubio(H00s, H01s, E)
        sig_b = (H01s.conj().T @ gs @ H01s)[0, 0]
        sig_a = complex(scalar_lead_sigma(E, t, eps0 + 2 * t))
        err = abs(sig_b - sig_a)
        maxerr = max(maxerr, err)
        print(f"{E:>7.2f} {sig_b.real:>12.6f}{sig_b.imag:>+11.6f}j "
              f"{sig_a.real:>12.6f}{sig_a.imag:>+11.6f}j {err:>10.1e}")
    print(f"  -> max|err| = {maxerr:.1e}  {'PASS' if maxerr < 1e-9 else 'FAIL'}\n")

    # DG-ALB bulk blocks --------------------------------------------------------
    H00, H01, n, herm, ref, sigma, dg_spec = dg_blocks(M=16)
    print(f"== DG-ALB blocks: n_keep={n}/element, sigma={sigma:.0f}, "
          f"assembled H Hermitian to {herm:.1e} ==")
    print(f"  ||H00||={np.abs(H00).max():.2e}  ||H01||={np.abs(H01).max():.2e}  "
          f"periodic DG spectrum range [{dg_spec[0]:.3f}, {dg_spec[-1]:.1f}]\n")

    # (2) BULK CONSISTENCY: DG-block Bloch bands must reproduce the periodic DG
    # spectrum EXACTLY (block-circulant), at the commensurate k = 2*pi*j/n_elem.
    ks = 2 * np.pi * np.arange(6) / 6
    bloch = []
    for k in ks:
        Hk = H00 + H01 * np.exp(1j * k) + H01.conj().T * np.exp(-1j * k)
        bloch.append(np.linalg.eigvalsh(Hk).real)
    bloch = np.sort(np.concatenate(bloch))
    berr = np.abs(bloch - dg_spec).max()
    print("== (2) bulk consistency: Bloch(H00,H01) vs periodic DG spectrum ==")
    print(f"  Bloch lowest 6:     {np.array2string(bloch[:6], precision=4)}")
    print(f"  periodic DG low 6:  {np.array2string(dg_spec[:6], precision=4)}")
    print(f"  -> max|Bloch - DG spectrum| = {berr:.1e}  "
          f"{'PASS' if berr < 1e-6 else 'FAIL (block extraction bug)'}\n")

    # (3) OPEN-BOUNDARY surface LDOS: positive, inside the bands ----------------
    emin, emax = ref[0] - 1.0, ref[8] + 1.0
    egrid = np.linspace(emin, emax, 40)
    ldos = np.array([-np.trace(sancho_rubio(H00, H01, E)).imag / np.pi for E in egrid])
    print("== (3) surface LDOS(E) = -Im tr g_surf / pi ==")
    print(f"  min LDOS = {ldos.min():.3e} (must be >= ~0), max = {ldos.max():.3e}")
    print(f"  -> {'PASS (no spurious negative DOS)' if ldos.min() > -1e-6 else 'FAIL'}\n")

    # (4) BOX-INDEPENDENCE: boundary LDOS vs device length ----------------------
    Eprobe = float(ref[3])                       # an occupied-band energy
    print("== (4) box-independence: boundary-layer LDOS vs device length ==")
    print(f"  probe E = {Eprobe:.4f}")
    vals = []
    for kdev in (2, 4, 8):
        v = device_ldos(H00, H01, kdev, Eprobe)
        vals.append(v)
        print(f"  kdev={kdev:>2}  boundary LDOS = {v:.6f}")
    spread = max(vals) - min(vals)
    print(f"  -> spread over device length = {spread:.1e}  "
          f"{'PASS (box-independent)' if spread < 1e-3 else 'CHECK'}")


if __name__ == "__main__":
    main()
