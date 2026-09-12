"""Magnon (spin-wave) band structures from a Heisenberg spin model via linear
spin-wave theory (LSWT).

This is the spin-wave layer that rides on top of the exchange couplings the
constrained-moment torque machinery extracts (:mod:`gradwave.postscf.spin_exchange`).
Given a classical spin Hamiltonian — isotropic Heisenberg J, Dzyaloshinskii-Moriya
D, symmetric-anisotropic Γ, and single-ion K — on a magnetic primitive cell, it
Holstein-Primakoff–linearizes about the (user-supplied) ordered ground state and
returns the magnon dispersion ω(q) along an ASE band path.

Convention
----------
The classical Hamiltonian summed over *ordered* pairs (each undirected bond
appears once per direction in the caller's list) is

    H = -½ Σ_{i,j,R} Ŝ_i(0)ᵀ 𝒥_ij(R) Ŝ_j(R)  -  Σ_i Ŝ_iᵀ A_i Ŝ_i ,          (1)

with Ŝ_i a spin operator of length S_i, 𝒥_ij(R) = J·I + Γ + 𝒟 the 3×3 exchange
tensor between sublattice i (home cell) and sublattice j in the cell displaced by
lattice vector R, and A_i = K_i n̂_i n̂_iᵀ the single-ion easy-axis term. Signs:

* **J > 0 is ferromagnetic** (aligning neighbours lowers the energy),
* **K > 0 is easy-axis** along n̂_i (a positive anisotropy gap),
* the DM vector enters as the antisymmetric part 𝒟_ab = Σ_c ε_abc D_c, so
  Ŝ_iᵀ 𝒟 Ŝ_j = D · (Ŝ_i × Ŝ_j).

For a one-sublattice collinear ferromagnet this collapses to the textbook
ω(q) = S[J(0) − J(q)] (+ gap), with J(q) = Σ_δ J_δ e^{iq·δ} over the full
coordination shell. The general (multi-sublattice, non-collinear, DM-canted)
case goes through the bosonic Bogoliubov / paraunitary (Colpa) diagonalization,
which is what is implemented; the collinear FM/AFM closed forms are recovered
exactly (the unit tests pin them).

The caller lists **every directed bond** (both (i,j,R) and its reverse (j,i,−R)
with the transposed tensor). :meth:`HeisenbergModel.from_shells` expands a shell
of symmetry-equivalent displacements into that directed list. Frequencies are
returned in the energy unit of the couplings (eV in ⇒ ħω in eV out); the task
layer converts to meV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.postscf.spin_exchange import _transverse_basis, _unit


class MagnonInstabilityError(ValueError):
    """The bosonic grand-dynamical matrix 𝓗(q) is not positive definite, so the
    Colpa/Bogoliubov diagonalization has no real-frequency solution. Physically
    this means the supplied state is **not a (meta)stable minimum** of the spin
    Hamiltonian — the most common causes are an incorrect ground-state ordering
    (wrong moment directions, e.g. FM directions on an antiferromagnet) or a
    coupling set truncated to too few shells to stabilize the order. The message
    reports the offending q."""


@dataclass(frozen=True)
class ExchangeBond:
    """One directed exchange bond of the model: the coupling from sublattice
    ``i`` (home cell) to sublattice ``j`` in the cell displaced by the integer
    lattice-vector triple ``R``. ``j_iso`` is the isotropic Heisenberg scalar
    (energy, J>0 ferromagnetic); ``dm`` the 3-vector Dzyaloshinskii-Moriya term
    (default zero); ``gamma`` the optional symmetric-traceless anisotropic-
    exchange 3×3 matrix (default zero)."""

    i: int
    j: int
    r: tuple[int, int, int]
    j_iso: float
    dm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamma: tuple[tuple[float, ...], ...] | None = None

    def tensor(self, dtype=torch.float64) -> torch.Tensor:
        """The 3×3 exchange tensor 𝒥 = J·I + Γ + 𝒟 (𝒟 the antisymmetric DM
        part with 𝒟_ab = Σ_c ε_abc D_c so Ŝ_iᵀ𝒟Ŝ_j = D·(Ŝ_i×Ŝ_j))."""
        j = self.j_iso * torch.eye(3, dtype=dtype)
        dx, dy, dz = self.dm
        j = j + torch.tensor(
            [[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]], dtype=dtype)
        if self.gamma is not None:
            j = j + torch.as_tensor(self.gamma, dtype=dtype)
        return j


@dataclass
class HeisenbergModel:
    """A classical Heisenberg spin model on a magnetic primitive cell — the
    numbers-in container for the LSWT dispersion.

    ``cell`` is the 3×3 lattice (Å; rows are the lattice vectors) whose Brillouin
    zone the q-path lives in. Per magnetic sublattice (``n_sub`` of them) the
    model carries the spin length ``spins`` S_i, the ordered-state moment
    direction ``moments`` ê_i (unit; default all +ẑ, i.e. a ferromagnet), the
    single-ion anisotropy ``anisotropy_k`` K_i (energy, easy-axis when >0), and
    the easy axis ``easy_axis`` n̂_i (default +ẑ). ``positions`` (fractional) are
    carried for reference/serialization; the dispersion eigenvalues are
    gauge-invariant to the intracell phase, so only the lattice vectors R enter
    the Fourier sum. ``bonds`` is the directed exchange-bond list (see the module
    docstring / :meth:`from_shells`)."""

    cell: np.ndarray  # (3,3) Å
    spins: np.ndarray  # (n_sub,)
    bonds: list[ExchangeBond]
    positions: np.ndarray | None = None  # (n_sub,3) fractional
    moments: np.ndarray | None = None  # (n_sub,3) unit; default +z
    anisotropy_k: np.ndarray | None = None  # (n_sub,) easy-axis single-ion K
    easy_axis: np.ndarray | None = None  # (n_sub,3) unit; default +z
    dtype: torch.dtype = field(default=torch.float64)

    def __post_init__(self):
        self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
        self.spins = np.asarray(self.spins, dtype=float).reshape(-1)
        n = self.n_sub
        if self.moments is None:
            self.moments = np.tile([0.0, 0.0, 1.0], (n, 1))
        self.moments = np.asarray(self.moments, dtype=float).reshape(n, 3)
        if self.anisotropy_k is None:
            self.anisotropy_k = np.zeros(n)
        self.anisotropy_k = np.asarray(self.anisotropy_k, dtype=float).reshape(-1)
        if self.easy_axis is None:
            self.easy_axis = np.tile([0.0, 0.0, 1.0], (n, 1))
        self.easy_axis = np.asarray(self.easy_axis, dtype=float).reshape(n, 3)
        for b in self.bonds:
            if not (0 <= b.i < n and 0 <= b.j < n):
                raise ValueError(
                    f"bond references sublattice {b.i}->{b.j} but the model has "
                    f"{n} sublattice(s) (spins length {n})")

    @property
    def n_sub(self) -> int:
        return len(self.spins)

    @classmethod
    def from_shells(
        cls,
        cell,
        spins,
        shells: list[dict],
        *,
        positions=None,
        moments=None,
        anisotropy_k=None,
        easy_axis=None,
    ) -> HeisenbergModel:
        """Build a model from *shells* of symmetry-equivalent bonds, expanding
        each shell into the directed bond list (both directions).

        Each shell is a mapping ``{"i", "j", "rs": [R, …], "j_iso", "dm"?,
        "gamma"?}``: the coupling ``j_iso`` (with optional ``dm``/``gamma``)
        applied to sublattice pair (i, j) for every displacement R in ``rs``. For
        every listed (i, j, R) the reverse (j, i, −R) with the transposed tensor
        is added automatically, so the caller lists each bond once per shell
        (e.g. the 8 bcc first neighbours as one shell of 8 R vectors — the
        reverses are generated). The DM part flips sign under reversal (its
        antisymmetry), which the transpose handles."""
        bonds: list[ExchangeBond] = []
        for sh in shells:
            i, j = int(sh["i"]), int(sh["j"])
            jiso = float(sh["j_iso"])
            dm = tuple(float(x) for x in sh.get("dm", (0.0, 0.0, 0.0)))
            gamma = sh.get("gamma")
            for r in sh["rs"]:
                r = tuple(int(x) for x in r)
                bonds.append(ExchangeBond(i, j, r, jiso, dm, gamma))  # type: ignore[arg-type]
                rev = tuple(-x for x in r)
                # reverse tensor is the transpose: DM negates, Γ transposes (Γ is
                # symmetric so unchanged), J unchanged.
                rev_dm = tuple(-x for x in dm)
                rev_gamma = (
                    None if gamma is None
                    else tuple(tuple(gamma[b][a] for b in range(3)) for a in range(3)))
                if (rev, j, i) != (r, i, j):  # skip a self-bond that is its own reverse
                    bonds.append(
                        ExchangeBond(j, i, rev, jiso, rev_dm, rev_gamma))  # type: ignore[arg-type]
        return cls(
            cell=cell, spins=spins, bonds=bonds, positions=positions,
            moments=moments, anisotropy_k=anisotropy_k, easy_axis=easy_axis)


@dataclass
class MagnonBandStructure:
    """Magnon dispersion along a q-path — the spin-wave analogue of
    :class:`gradwave.postscf.bands.BandStructure`.

    ``frequencies`` are the magnon energies ħω in **meV**, shape (nq, n_branch)
    with n_branch = n_sub. ``qpts_frac`` are the fractional reciprocal-space
    q-points; ``labels`` the (index, symbol) high-symmetry markers and ``x`` the
    linear path coordinate for plotting (both from the ASE band path)."""

    qpts_frac: np.ndarray  # (nq, 3)
    frequencies: np.ndarray  # (nq, n_branch) [meV]
    labels: list[tuple[int, str]] | None = None
    x: np.ndarray | None = None


def _local_frames(model: HeisenbergModel) -> torch.Tensor:
    """Complex transverse vectors u_i = ê1_i + i ê2_i (n_sub, 3), where
    (ê1, ê2, ê_moment) is a right-handed orthonormal triad — the Holstein-
    Primakoff local frame for each sublattice's moment direction."""
    n = model.n_sub
    u = torch.zeros(n, 3, dtype=torch.complex128)
    for a in range(n):
        eta = _unit(torch.as_tensor(model.moments[a], dtype=torch.float64))
        e1, e2 = _transverse_basis(eta)  # e1 × e2 = eta (right-handed)
        u[a] = e1.to(torch.complex128) + 1j * e2.to(torch.complex128)
    return u


def _coupling_fourier(model: HeisenbergModel, q_frac: np.ndarray) -> torch.Tensor:
    """Fourier-transformed exchange tensors Q_ij(q) = Σ_R 𝒥_ij(R) e^{i 2π q·R},
    shape (n_sub, n_sub, 3, 3) complex. q_frac is in fractional reciprocal
    coordinates, so q·R = 2π (q_frac · R_int)."""
    n = model.n_sub
    q = torch.zeros(n, n, 3, 3, dtype=torch.complex128)
    qf = np.asarray(q_frac, dtype=float)
    for b in model.bonds:
        phase = np.exp(2j * np.pi * float(qf @ np.asarray(b.r, dtype=float)))
        q[b.i, b.j] += b.tensor(torch.float64).to(torch.complex128) * phase
    return q


def _grand_matrix(
    model: HeisenbergModel, u: torch.Tensor, q_frac: np.ndarray
) -> torch.Tensor:
    """Assemble the 2N×2N bosonic grand-dynamical matrix 𝓗(q) of the LSWT
    Hamiltonian ½ Xᵀ𝓗X, X = (a_{1,q}…a_{N,q}, a†_{1,-q}…a†_{N,-q}).

    Blocks (derived from the Holstein-Primakoff substitution of eq. (1)):

        A_ij(q) = -½ √(S_i S_j) u_iᵀ Q_ij(q) u_j*        (particle hopping)
        B_ij(q) = -½ √(S_i S_j) u_iᵀ Q_ij(q) u_j          (pairing)
        C_i     =  Σ_l S_l ê_iᵀ Q_il(0) ê_l + 2 S_i K_i (ê_i·n̂_i)²   (molecular field)

    𝓗(q) = [[A(q)+diag(C),   B(q)          ],
            [B(q)†,           conj(A(−q))+diag(C)]]
    (u_iᵀ … u_j* here means Σ_ab conj(u_i^a) 𝒥^ab u_j^b — the Hermitian form.)"""
    s = torch.as_tensor(model.spins, dtype=torch.float64)
    sqrt_ss = torch.sqrt(torch.outer(s, s)).to(torch.complex128)  # (n,n)
    uc = u.conj()

    qp = _coupling_fourier(model, q_frac)
    qm = _coupling_fourier(model, -np.asarray(q_frac, dtype=float))
    q0 = _coupling_fourier(model, np.zeros(3))

    # A_ij = -1/2 sqrt(S_iS_j) sum_ab conj(u_i^a) Q^ab u_j^b
    #      = -1/2 sqrt(S_iS_j) (uc_i · (Q_ij @ u_j))
    def _a_block(qmat: torch.Tensor) -> torch.Tensor:
        # qmat: (n,n,3,3); u: (n,3)
        qu = torch.einsum("ijab,jb->ija", qmat, u)          # (n,n,3)
        a = torch.einsum("ia,ija->ij", uc, qu)              # conj(u_i)·(Q u_j)
        return -0.5 * sqrt_ss * a

    a_plus = _a_block(qp)
    a_minus = _a_block(qm)

    # B_ij = -1/2 sqrt(S_iS_j) sum_ab conj(u_i^a) Q^ab conj(u_j^b)
    qcu = torch.einsum("ijab,jb->ija", qp, uc)
    b_block = -0.5 * sqrt_ss * torch.einsum("ia,ija->ij", uc, qcu)

    # molecular field C_i (real diagonal): sum_l S_l eta_i . Q0_il . eta_l
    eta = torch.as_tensor(np.asarray(model.moments), dtype=torch.float64)
    eta = eta / torch.linalg.norm(eta, dim=1, keepdim=True)
    q0r = q0.real  # Q0 is real for a Hermitian coupling set
    # eta_i^a Q0_ijab eta_j^b -> (n,n)
    field = torch.einsum("ia,ijab,jb->ij", eta, q0r, eta)
    c = (torch.as_tensor(model.spins, dtype=torch.float64)[None, :] * field).sum(dim=1)
    # single-ion: 2 S_i K_i (eta_i . n_i)^2
    nax = torch.as_tensor(np.asarray(model.easy_axis), dtype=torch.float64)
    nax = nax / torch.linalg.norm(nax, dim=1, keepdim=True)
    k = torch.as_tensor(model.anisotropy_k, dtype=torch.float64)
    proj = torch.einsum("ia,ia->i", eta, nax) ** 2
    c = c + 2.0 * torch.as_tensor(model.spins, dtype=torch.float64) * k * proj
    diag_c = torch.diag(c.to(torch.complex128))

    top_left = a_plus + diag_c
    bot_right = a_minus.conj() + diag_c
    top = torch.cat([top_left, b_block], dim=1)
    bot = torch.cat([b_block.conj().transpose(0, 1), bot_right], dim=1)
    h = torch.cat([top, bot], dim=0)
    # defensive Hermitization: kills round-off asymmetry; preserves DM
    # nonreciprocity (each 𝓗(q) is Hermitian; DM makes 𝓗(q) ≠ 𝓗(-q)).
    return 0.5 * (h + h.conj().transpose(0, 1))


def _colpa(h: torch.Tensor, *, imag_tol: float = 1e-6) -> torch.Tensor:
    """Bogoliubov (Colpa) diagonalization of a bosonic grand matrix 𝓗 (2N×2N,
    Hermitian): return the N magnon frequencies (ascending, ≥ 0).

    Colpa's method: Cholesky 𝓗 = K†K (requires 𝓗 positive definite), then the
    eigenvalues of K g K† (g = diag(I_N, −I_N)) come in ± pairs whose positive
    members are the magnon energies. At a Goldstone point 𝓗 is only positive
    *semi*-definite and Cholesky fails; there we fall back to the eigenvalues of
    g𝓗 (which are still real, including the exact zero mode). A genuinely
    non-positive-definite 𝓗 — an unstable ground state — yields *complex*
    eigenvalues of g𝓗, which raise :class:`MagnonInstabilityError`."""
    n2 = h.shape[0]
    n = n2 // 2
    g = torch.diag(torch.cat([
        torch.ones(n, dtype=h.dtype), -torch.ones(n, dtype=h.dtype)]))

    # Positive-(semi)definiteness is the Colpa precondition and the physical
    # stability test in one: a strictly-negative eigenvalue of 𝓗 means the
    # ordered state is a saddle/maximum of the spin Hamiltonian (unstable),
    # while a zero eigenvalue is a Goldstone mode (gapless but stable).
    h_evals = torch.linalg.eigvalsh(h)
    scale = float(h_evals.abs().max()) + 1e-30
    if float(h_evals.min()) < -imag_tol * scale:
        raise MagnonInstabilityError(
            "the magnon grand matrix 𝓗(q) is not positive definite: the supplied "
            "ordered state is not a stable minimum of the spin Hamiltonian. "
            "Likely causes — wrong ground-state moment directions (e.g. "
            "ferromagnetic directions on an antiferromagnet), or a coupling set "
            "truncated to too few shells. "
            f"min eig(𝓗)/scale = {float(h_evals.min()) / scale:.2e}")

    if float(h_evals.min()) > 1e-9 * scale:
        # strictly positive definite -> Colpa via Cholesky (accurate + stable)
        ell = torch.linalg.cholesky(h)          # h = L L†, L lower
        kmat = ell.conj().transpose(0, 1)       # upper K, h = K† K
        w = kmat @ g @ kmat.conj().transpose(0, 1)
        w = 0.5 * (w + w.conj().transpose(0, 1))
        evals = torch.linalg.eigvalsh(w)        # real, ascending, ± paired
        return evals[n:].clamp_min(0.0)         # top N = +ω, ascending
    # positive semidefinite (Goldstone): Cholesky fails, so read ±ω off g𝓗
    ev = torch.linalg.eigvals(g @ h)
    real = torch.sort(ev.real).values
    return real[n:].clamp_min(0.0)


def magnon_dispersion(
    model: HeisenbergModel, qpts_frac: np.ndarray
) -> np.ndarray:
    """Magnon frequencies ħω(q) at the given fractional q-points.

    Returns an (nq, n_sub) array in the **energy unit of the model couplings**
    (eV in ⇒ eV out). Each row is the branch energies (ascending). Raises
    :class:`MagnonInstabilityError` (naming the q) if the ordered state is
    unstable at some q."""
    u = _local_frames(model)
    qpts = np.asarray(qpts_frac, dtype=float).reshape(-1, 3)
    out = np.empty((len(qpts), model.n_sub))
    for iq, q in enumerate(qpts):
        h = _grand_matrix(model, u, q)
        try:
            out[iq] = _colpa(h).cpu().numpy()
        except MagnonInstabilityError as exc:
            raise MagnonInstabilityError(f"{exc} (at q = {q.tolist()})") from None
    return out


def magnon_bands(
    model: HeisenbergModel, path: str = "", npoints: int = 200
) -> MagnonBandStructure:
    """Magnon dispersion along an ASE band path of the magnetic primitive cell.

    Mirrors :func:`gradwave.postscf.bands.bands_along_ase_path`: the path rides
    ``model.cell`` (a special-point string, or the lattice default when empty),
    with the linear coordinate ``x`` and high-symmetry ``labels`` for plotting.
    Frequencies are returned in **meV**."""
    from ase.cell import Cell

    bp = Cell(model.cell).bandpath(path=path or None, npoints=npoints)
    freqs_ev = magnon_dispersion(model, bp.kpts)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    return MagnonBandStructure(
        qpts_frac=np.asarray(bp.kpts),
        frequencies=freqs_ev * 1.0e3,  # eV -> meV
        labels=list(zip(xticks.tolist(), list(xlabels), strict=True)),
        x=np.asarray(x),
    )


def spin_wave_stiffness(
    model: HeisenbergModel, *, direction=(1.0, 0.0, 0.0), qmax_frac: float = 0.02,
    npoints: int = 12,
) -> float:
    """Ferromagnetic spin-wave stiffness D from a small-q parabolic fit
    ω(q) ≈ D q² of the acoustic branch along ``direction`` (a Cartesian
    reciprocal direction, normalized here).

    Returns D in **meV·Å²** (couplings in eV, cell in Å): fits ω[meV] vs
    |q|²[Å⁻²] through the origin over |q| up to ``qmax_frac`` of the reciprocal
    lattice. Meaningful only for a ferromagnetic ground state (a single acoustic
    branch through ω=0)."""
    # cartesian reciprocal lattice (Å⁻¹), rows b_i, with a_i·b_j = 2π δ_ij
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(model.cell, dtype=float)).T
    inv_recip = np.linalg.inv(recip)  # cartesian q (Å⁻¹) -> fractional q
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    # sweep |q| in Å⁻¹ up to qmax_frac × |b1| (a small fraction of the zone)
    qmax_cart = qmax_frac * float(np.linalg.norm(recip[0]))
    qmags = np.linspace(0.0, qmax_cart, npoints + 1)[1:]
    q2 = []
    w = []
    for qm in qmags:
        qfrac = (qm * d) @ inv_recip
        omega = magnon_dispersion(model, qfrac[None, :])[0]
        q2.append(qm**2)
        w.append(float(omega.min()) * 1.0e3)  # meV
    q2 = np.asarray(q2)
    w = np.asarray(w)
    # least-squares slope through the origin: D = (Σ q² ω) / (Σ q⁴)
    return float((q2 @ w) / (q2 @ q2))
