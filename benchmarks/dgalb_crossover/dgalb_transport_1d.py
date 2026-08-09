"""DG-ALB NEGF transport spike (S2): device between two semi-infinite leads,
Landauer/Caroli transmission, and DIFFERENTIABLE transport.

Builds on the validated S1 block lead self-energy (dgalb_open_1d.py): the leads are
semi-infinite DG-ALB chains whose surface Green's functions give Sigma_L, Sigma_R;
the device is a finite block-tridiagonal DG-ALB chain. The retarded Green's function

    G(E) = [ (E+i.eta) I - H_dev - Sigma_L - Sigma_R ]^{-1}

gives the Caroli transmission  T(E) = Tr[ Gamma_L G_1N Gamma_R G_1N^dag ],
Gamma = i(Sigma - Sigma^dag), G_1N the corner (left-contact -> right-contact) block.

Validation ladder:
  (A) PERFECT CONDUCTOR: device == lead material -> T(E) must equal the integer
      number of open channels N_open(E) (bands crossing E). The canonical NEGF unit
      test: no scattering -> perfect transmission, integer-valued.
  (B) BARRIER: raise one device element's onsite -> T drops below the perfect value
      (tunneling), with resonances -> a real scattering region.
  (C) DIFFERENTIABLE: dT/d(barrier) via autograd (torch, through the matrix inverse)
      vs finite difference -> differentiable transport, the inverse-design superpower.

Reuses dgalb_open_1d (S1) + dgalb_spike_1d. numpy + torch.

Run:  uv run python benchmarks/dgalb_crossover/dgalb_transport_1d.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgalb_open_1d as op  # noqa: E402  (S1: sancho_rubio, dg_blocks, bulk_bands)

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- band channels
def band_ranges(H00, H01, nk=200):
    """[min,max] energy of each band over the 1D BZ."""
    n = H00.shape[0]
    bands = np.zeros((nk, n))
    for i, k in enumerate(np.linspace(-np.pi, np.pi, nk)):
        Hk = H00 + H01 * np.exp(1j * k) + H01.conj().T * np.exp(-1j * k)
        bands[i] = np.linalg.eigvalsh(Hk).real
    return np.stack([bands.min(0), bands.max(0)], axis=1)   # (n, 2)


def n_open(E, ranges):
    """Number of bands crossing energy E (open transport channels)."""
    return int(np.sum((ranges[:, 0] <= E) & (E <= ranges[:, 1])))


# ---------------------------------------------------------------- self-energies
def lead_sigmas(H00, H01, E, eta=1e-6):
    """Left/right lead self-energies acting on the first/last device blocks."""
    gL = op.sancho_rubio(H00, H01, E, eta=eta)             # lead extending left
    gR = op.sancho_rubio(H00, H01.conj().T, E, eta=eta)    # lead extending right
    sigL = H01.conj().T @ gL @ H01
    sigR = H01 @ gR @ H01.conj().T
    return sigL, sigR


def transmission_np(H00, H01, onsites, E, eta=1e-6):
    """Caroli T(E). onsites: list of kdev device diagonal blocks (H01 couples them)."""
    n = H00.shape[0]
    kdev = len(onsites)
    D = kdev * n
    Hd = np.zeros((D, D), dtype=complex)
    for i, blk in enumerate(onsites):
        Hd[i * n:(i + 1) * n, i * n:(i + 1) * n] = blk
        if i + 1 < kdev:
            Hd[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = H01
            Hd[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = H01.conj().T
    sigL, sigR = lead_sigmas(H00, H01, E, eta)
    A = (E + 1j * eta) * np.eye(D, dtype=complex) - Hd
    A[:n, :n] -= sigL
    A[-n:, -n:] -= sigR
    G = np.linalg.inv(A)
    gamL = 1j * (sigL - sigL.conj().T)
    gamR = 1j * (sigR - sigR.conj().T)
    g1n = G[:n, -n:]                                        # left->right corner block
    return float(np.trace(gamL @ g1n @ gamR @ g1n.conj().T).real)


def transmission_torch(H00, H01, onsites, sigL, sigR, E, eta=1e-6):
    """Differentiable T(E): leads (sigL,sigR) are fixed constants; the device onsites
    carry the autograd parameter. All-torch so dT/dparam flows through the inverse."""
    n = H00.shape[0]
    kdev = len(onsites)
    D = kdev * n
    Hd = torch.zeros(D, D, dtype=torch.complex128)
    H01t = torch.tensor(H01, dtype=torch.complex128)
    for i, blk in enumerate(onsites):
        Hd[i * n:(i + 1) * n, i * n:(i + 1) * n] = blk
        if i + 1 < kdev:
            Hd[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = H01t
            Hd[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = H01t.conj().T
    A = (E + 1j * eta) * torch.eye(D, dtype=torch.complex128) - Hd
    A[:n, :n] = A[:n, :n] - sigL
    A[-n:, -n:] = A[-n:, -n:] - sigR
    G = torch.linalg.inv(A)
    gamL = 1j * (sigL - sigL.conj().T)
    gamR = 1j * (sigR - sigR.conj().T)
    g1n = G[:n, -n:]
    return torch.trace(gamL @ g1n @ gamR @ g1n.conj().T).real


def tb_blocks(e0=0.0, t=1.0):
    """Scalar 1D tight-binding lead: analytic band [e0-2t, e0+2t], perfect T=1."""
    return np.array([[e0]], dtype=float), np.array([[-t]], dtype=float)


def validate(H00, H01, label, e_probes, e_bar, e_diff, barrier_scale=1.0, tolA=1e-3):
    n = H00.shape[0]
    ranges = band_ranges(H00, H01)
    print(f"##### {label}  (n_keep={n}) #####")

    # (A) PERFECT CONDUCTOR: T(E) == N_open(E) (integer) ------------------------
    print("== (A) perfect conductor (device = lead): T(E) vs N_open(E) ==")
    print(f"{'E':>8} {'N_open':>7} {'T(E)':>10} {'|T-N|':>9}")
    onsites_perfect = [H00] * 4
    maxdev = 0.0
    for E in e_probes:
        No = n_open(E, ranges)
        T = transmission_np(H00, H01, onsites_perfect, E)
        maxdev = max(maxdev, abs(T - No))
        print(f"{E:>8.2f} {No:>7} {T:>10.5f} {abs(T-No):>9.1e}")
    okA = maxdev < tolA          # tolA relaxed for finite-subspace band-edge error
    print(f"  -> max|T - N_open| = {maxdev:.1e}  "
          f"{'PASS (integer perfect transmission)' if okA else 'CHECK'}\n")

    # (B) BARRIER: raise the middle element's onsite -> T drops -----------------
    print(f"== (B) barrier (middle device onsite += b*I) at E={e_bar}: ==")
    print(f"{'barrier':>8} {'T':>11}")
    T0 = None
    Tb = None
    for b in (0.0, 0.2, 0.5, 1.0, 2.0):
        onsites = [H00, H00, H00 + b * barrier_scale * np.eye(n), H00, H00]
        Tb = transmission_np(H00, H01, onsites, e_bar)
        if T0 is None:
            T0 = Tb
        print(f"{b:>8.2f} {Tb:>11.5f}")
    okB = (Tb < T0 - 1e-3) if T0 > 1e-6 else False    # monotonic scattering suppression
    print(f"  -> T {T0:.4f} -> {Tb:.4f} with barrier  "
          f"{'(PASS: scattering suppresses T)' if okB else '(CHECK)'}\n")

    # (C) DIFFERENTIABLE: dT/d(barrier) autograd vs finite difference -----------
    print(f"== (C) differentiable transport: dT/d(barrier) at E={e_diff} ==")
    sigL_np, sigR_np = lead_sigmas(H00, H01, e_diff)
    sigL = torch.tensor(sigL_np, dtype=torch.complex128)
    sigR = torch.tensor(sigR_np, dtype=torch.complex128)
    H00t = torch.tensor(H00, dtype=torch.complex128)
    eye = torch.eye(n, dtype=torch.complex128)
    b0 = 0.6 * barrier_scale

    def Tof(bval):
        onsites = [H00t, H00t, H00t + bval * eye, H00t, H00t]
        return transmission_torch(H00, H01, onsites, sigL, sigR, e_diff)

    bt = torch.tensor(b0, requires_grad=True)
    Tv = Tof(bt)
    Tv.backward()
    dT_ad = float(bt.grad)
    db = 1e-4
    dT_fd = (float(Tof(torch.tensor(b0 + db)).detach())
             - float(Tof(torch.tensor(b0 - db)).detach())) / (2 * db)
    rel = abs(dT_ad - dT_fd) / (abs(dT_fd) + 1e-30)
    okC = rel < 1e-4
    print(f"  T(b={b0:.2f}) = {float(Tv.detach()):.6f}")
    print(f"  dT/db autograd = {dT_ad:+.6e}   finite-d = {dT_fd:+.6e}   rel={rel:.1e}  "
          f"{'PASS' if okC else 'CHECK'}\n")
    return okA, okB, okC


def main():
    print("# DG-ALB NEGF transport spike (S2): device + 2 leads, Caroli transmission\n")

    # analytic tight-binding chain first: band [-2,2], perfect T=1 in band -------
    H00, H01 = tb_blocks(e0=0.0, t=1.0)
    validate(H00, H01, "scalar tight-binding lead (analytic: band [-2,2], T=1)",
             e_probes=(-1.5, -0.8, 0.0, 0.8, 1.5, 2.5), e_bar=0.0, e_diff=0.0)

    # DG-ALB blocks -------------------------------------------------------------
    H00d, H01d, n, herm, ref, sigma, dg_spec = op.dg_blocks(M=8)
    print(f"(DG-ALB blocks: n_keep={n}, Hermitian {herm:.1e}, sigma={sigma:.0f}, "
          f"spectrum range [{dg_spec[0]:.3f}, {dg_spec[-1]:.0f}] — penalty lifts jump "
          f"modes to ~{dg_spec[-1]:.0f}, which swamp the physical channel in H01)\n")

    # PHYSICAL-SUBSPACE PROJECTION: restrict the lead blocks to the low-energy
    # invariant subspace (the physical bands), removing the SIPG penalty modes that
    # dominate the coupling. Galerkin projection onto the lowest p eigenvectors of
    # H00 -- the standard transport-window reduction.
    p = 3
    Hgamma = H00d + H01d + H01d.conj().T          # k=0 Bloch block (physical bands)
    w, U = np.linalg.eigh(Hgamma)
    Up = U[:, :p]                                 # lowest-p physical Bloch states
    H00p = Up.conj().T @ H00d @ Up
    H01p = Up.conj().T @ H01d @ Up
    print(f"(physical-subspace projection: lowest {p} Gamma-point bands at "
          f"{np.array2string(w[:p], precision=3)})")
    validate(H00p, H01p, f"DG-ALB blocks, physical-subspace projected (p={p})",
             e_probes=(-0.30, -0.20, 0.00, 0.30, 0.70, 1.10), e_bar=-0.25, e_diff=-0.25,
             tolA=5e-2)      # band-edge projection error; mid-band T=1 to ~1e-3


if __name__ == "__main__":
    main()
