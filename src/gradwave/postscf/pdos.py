"""Projected density of states (Layer C post-processing).

Projects the converged Kohn-Sham states onto the pseudo-atomic orbitals
(PP_PSWFC) to resolve the DOS by atom, angular momentum, and magnetic quantum
number. The atomic-orbital projectors reuse the Kleinman-Bylander structure of
core/hubbard.py,

    phi_{a,nlm}(k+G) = (4 pi / sqrt(Omega)) (-i)^l Y_lm(k+G^) F_nl(|k+G|)
                       e^{-i (k+G).tau_a},   F_nl(q) = sbt(l, r^2 R_nl, r, rab, q)

and the projection is Loewdin-orthonormalized so the per-state weights obey the
sum rule up to the plane-wave truncation, which the spilling parameter reports.

For USPP/PAW the overlap is not the identity, so the projection uses the S-metric
<phi|S|psi> and the Loewdin overlap <phi|S|phi>, with S = 1 + sum_ij |beta_i> q_ij
<beta_j| built from the same augmentation charges q_ij the SCF uses.

Coverage here: norm-conserving and USPP/PAW, nspin=1 and 2. The noncollinear/SOC
projections extend this in the same module.

Reference: D. Sanchez-Portal et al., the Loewdin population analysis behind QE's
projwfc.x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.constants import MINUS_I_POW as _MINUS_I_POW
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.radial import sbt
from gradwave.scf.loop import SCFResult

_M_LABELS = {0: [""], 1: ["z", "x", "y"],  # real-harmonic order of ylm_all
             2: ["z2", "xz", "yz", "x2-y2", "xy"],
             3: ["z3", "xz2", "yz2", "zx2-zy2", "xyz", "xx2-3yy2", "3xx2-yy2"]}


@dataclass
class AOColumn:
    """One atomic-orbital projector column."""

    atom: int
    species: int
    label: str        # e.g. "3D"
    l: int
    m: int            # 0..2l, real-harmonic index


@dataclass
class ProjectedDOS:
    energy_eV: np.ndarray            # (npoints,)
    total: np.ndarray                # (npoints,) or (2, npoints) for nspin=2
    groups: dict                     # group label -> same shape as total
    spilling: float                  # fraction of KS weight outside the AO span
    fermi_eV: float | None
    nspin: int
    group_by: str

    def to_dict(self) -> dict:
        """JSON-ready block (lists, not arrays); the parsing target for the
        analysis layer."""
        def _col(a):
            return np.asarray(a).tolist()
        return {
            "energy_eV": _col(self.energy_eV),
            "total": _col(self.total),
            "groups": {k: _col(v) for k, v in self.groups.items()},
            "spilling": self.spilling,
            "fermi_eV": self.fermi_eV,
            "nspin": self.nspin,
            "group_by": self.group_by,
        }


@dataclass
class NoncollinearPDOS:
    """Projected DOS of a noncollinear spinor SCF. `charge` is the atom/orbital-
    resolved DOS n(E); `m_x/m_y/m_z` are the spin-texture components, the Pauli
    decomposition <phi~|sigma|phi~> of the same projected amplitudes, so a group's
    magnetization density of states is (m_x, m_y, m_z) and its magnitude names the
    local spin axis at each energy."""

    energy_eV: np.ndarray            # (npoints,)
    total_charge: np.ndarray         # (npoints,)
    charge: dict                     # group label -> (npoints,)
    m_x: dict                        # group label -> (npoints,)
    m_y: dict
    m_z: dict
    spilling: float
    fermi_eV: float | None
    group_by: str

    def to_dict(self) -> dict:
        def _col(a):
            return np.asarray(a).tolist()

        def _grp(d):
            return {k: _col(v) for k, v in d.items()}
        return {
            "energy_eV": _col(self.energy_eV),
            "total_charge": _col(self.total_charge),
            "charge": _grp(self.charge),
            "m_x": _grp(self.m_x), "m_y": _grp(self.m_y), "m_z": _grp(self.m_z),
            "spilling": self.spilling,
            "fermi_eV": self.fermi_eV,
            "noncollinear": True,
            "group_by": self.group_by,
        }


def _broaden(grid, energies, per_state, width):
    """Gaussian-broadened spectral sum. energies (nstate,), per_state (nstate,)
    already carries the k-weight, spin degeneracy, and orbital weight."""
    inv = 1.0 / (width * math.sqrt(2 * math.pi))
    return (np.exp(-0.5 * ((grid[:, None] - energies[None, :]) / width) ** 2)
            * inv * per_state[None, :]).sum(axis=1)


def spectral_grid(all_e, width, npoints, window=None):
    """(window, energy grid) for DOS/COHP broadening. When `window` is None it
    defaults to the eigenvalue range padded by 10*width on each side — far enough
    that a Gaussian of that width has decayed. Shared by the DOS functions here
    and cohp._finalize so the padding rule lives in one place."""
    if window is None:
        window = (all_e.min() - 10 * width, all_e.max() + 10 * width)
    return window, np.linspace(window[0], window[1], npoints)


def _is_uspp_system(system) -> bool:
    """USPP/PAW systems carry the augmentation weights; NC systems carry .upfs."""
    return hasattr(system, "paws") and hasattr(system, "q_full")


def _species_orbitals(system, sp):
    """(pseudopotential, PP_PSWFC atomic orbitals) for species `sp`. The radial
    tables live on .pswfc for norm-conserving UPFData and on .chi for PAWData,
    but both are AtomicOrbital(l, label, rchi=r·R_nl) with matching .r/.rab."""
    if _is_uspp_system(system):
        pp = system.paws[sp]
        return pp, getattr(pp, "chi", ())
    pp = system.upfs[sp]
    return pp, getattr(pp, "pswfc", ())


def _atomic_columns(system) -> list[AOColumn]:
    """Every PP_PSWFC orbital of every atom, expanded over m."""
    cols = []
    for a, sp in enumerate(system.species_of_atom):
        pp, orbs = _species_orbitals(system, sp)
        if not orbs:
            raise ValueError(
                f"{pp.element}: the pseudopotential carries no PP_PSWFC atomic "
                "orbitals, so a projected DOS is not available (SG15 ONCV omits "
                "them; use a PseudoDojo or psl pseudo)")
        for o in orbs:
            for m in range(2 * o.l + 1):
                cols.append(AOColumn(a, sp, o.label, o.l, m))
    return cols


def _ao_projectors_k(system, sph, cols, device):
    """AO projectors q (nproj, npw) on one G-sphere, phased at the positions."""
    vol = system.grid.volume
    kpg = sph.kpg.to(device)
    npw = sph.npw
    qmag = np.sqrt(sph.kpg2.cpu().numpy())
    l_max = max(c.l for c in cols)
    y = ylm_all(l_max, kpg)  # (npw, (l_max+1)^2)
    # radial form factors F_nl(q), cached per (species, orbital label)
    fcache: dict[tuple, torch.Tensor] = {}
    for sp in set(system.species_of_atom):
        u, orbs = _species_orbitals(system, sp)
        for o in orbs:
            key = (sp, o.label)
            if key not in fcache:
                fcache[key] = torch.as_tensor(
                    sbt(o.l, o.rchi * u.r, u.r, u.rab, qmag),
                    dtype=RDTYPE, device=device)
    # phases e^{-i(k+G).tau_a}
    phase_arg = kpg @ system.positions.to(device).T  # (npw, na)
    phase = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

    q = torch.zeros(len(cols), npw, dtype=CDTYPE, device=device)
    for p, c in enumerate(cols):
        pref = (4.0 * math.pi / math.sqrt(vol)) * _MINUS_I_POW[c.l]
        yl = y[:, c.l * c.l + c.m]
        q[p] = pref * (fcache[(c.species, c.label)] * yl).to(CDTYPE) * phase[:, c.atom]
    return q


def o_inv_sqrt(overlap, floor=1e-8):
    """O^{-1/2} (Hermitian) of an AO overlap O, near-singular modes clamped to
    `floor`. The single definition shared by the Loewdin projection here and the
    COHP operator route in cohp.py, so the two cannot drift."""
    w, v = torch.linalg.eigh(overlap)
    w = w.clamp_min(floor)
    return (v * w.rsqrt()) @ v.conj().T                # O^{-1/2}, Hermitian


def _lowdin_project(becp, overlap, floor=1e-8):
    """Loewdin-orthonormalized amplitudes <phi~_p|psi_b> = (<phi_p|psi_b> O^{-1/2})
    from raw becp (nb, nproj) and the AO overlap O = <phi_i|phi_j> (nproj, nproj).
    Returns the complex amplitudes so a caller can form |.|^2 (populations) or the
    cross terms proj_up* proj_dn (the noncollinear spin texture)."""
    # <phi~_p|psi> = sum_q (O^{-1/2})_pq <phi_q|psi> = (becp @ O^{-1/2 T}). Since
    # O^{-1/2} is Hermitian, O^{-1/2 T} = conj(O^{-1/2}); the conjugate matters
    # whenever the AO overlap is complex (general k with Bloch phases) — without
    # it the captured weight exceeds 1 (negative spilling) off Gamma.
    return becp @ o_inv_sqrt(overlap, floor).conj()    # (nb, nproj), complex


def _lowdin_weights(becp, overlap, floor=1e-8):
    """Loewdin-orthonormalized populations |<phi~_p|psi_b>|^2 (nb, nproj)."""
    proj = _lowdin_project(becp, overlap, floor)
    return proj.real ** 2 + proj.imag ** 2


def split_spinor(c, npw, m_pw):
    """Up/down plane-wave blocks (cu, cd) of a spinor coefficient block
    c (nb, 2*m_pw): the up component occupies [:npw], the down component starts
    at the fixed padding offset m_pw. Shared by the noncollinear/SOC PDOS and
    COHP projection loops."""
    return c[:, :npw], c[:, m_pw:m_pw + npw]


def spinor_scalar_amplitudes(system, sph, cols, cu, cd, device):
    """Loewdin amplitudes (pu, pd) of a spinor state's up/down components on the
    SCALAR AO set, sharing one spatial overlap. Complex (nb, nproj) each — the
    caller forms |.|^2 populations or the pu* pd cross term (spin texture).
    Setup only (no autograd path)."""
    q = _ao_projectors_k(system, sph, cols, device)
    overlap = torch.einsum("ig,jg->ij", q.conj(), q)
    pu = _lowdin_project(torch.einsum("bg,pg->bp", cu, q.conj()), overlap)
    pd = _lowdin_project(torch.einsum("bg,pg->bp", cd, q.conj()), overlap)
    return pu, pd


def spinor_jmj_amplitudes(system, sph, cols, cu, cd, device):
    """Loewdin amplitudes (nb, nproj) on the |l j mj> spin-angular AO set; the
    becp and AO overlap are spin-summed over the two spinor components. Complex —
    the caller forms |.|^2 (SOC populations) or uses the amplitudes directly (the
    band-limited COHP). Setup only (no autograd path)."""
    qu, qd = _ao_spinor_projectors_k(system, sph, cols, device)
    becp = (torch.einsum("bg,pg->bp", cu, qu.conj())
            + torch.einsum("bg,pg->bp", cd, qd.conj()))
    overlap = (torch.einsum("pg,qg->pq", qu.conj(), qu)
               + torch.einsum("pg,qg->pq", qd.conj(), qd))
    return _lowdin_project(becp, overlap)


def _nc_weights_k(system, sph, ik, c, cols, device):
    """Norm-conserving Löwdin weights (nb, nproj); overlap is the bare AO Gram."""
    q = _ao_projectors_k(system, sph, cols, device)     # (nproj, npw)
    becp = torch.einsum("bg,pg->bp", c, q.conj())        # <phi_p|psi_b>
    overlap = torch.einsum("ig,jg->ij", q.conj(), q)     # <phi_i|phi_j>
    return _lowdin_weights(becp, overlap).cpu().numpy()


def _uspp_weights_k(system, sph, ik, c, cols, device):
    """USPP/PAW Löwdin weights with the S-metric. becp = <phi|S|psi> and the
    overlap = <phi|S|phi>, where the augmentation S = 1 + sum_ij |beta_i> q_ij
    <beta_j| reuses the SCF's m-expanded beta projectors and charges q_full."""
    from gradwave.core.hamiltonian import projectors
    q = _ao_projectors_k(system, sph, cols, device)             # (nproj, npw)
    pbeta = projectors(system.proj_data[ik], system.positions).to(device)  # (nb_i, npw)
    qf = system.q_full.to(device).to(CDTYPE)                    # (nb_i, nb_i)
    becp_bare = torch.einsum("bg,pg->bp", c, q.conj())          # <phi_p|psi_b>
    beta_psi = torch.einsum("bg,ig->bi", c, pbeta.conj())       # <beta_i|psi_b>
    phi_beta = torch.einsum("pg,ig->pi", q.conj(), pbeta)       # <phi_p|beta_i>
    pq = phi_beta @ qf                                          # (nproj, nb_i)
    becp = becp_bare + torch.einsum("pj,bj->bp", pq, beta_psi)  # <phi|S|psi>
    overlap = torch.einsum("ig,jg->ij", q.conj(), q) + pq @ phi_beta.conj().T
    return _lowdin_weights(becp, overlap).cpu().numpy()


def _group_key(col: AOColumn, group_by: str):
    if group_by == "total":
        return "total"
    if group_by == "atom":
        return f"atom{col.atom + 1}"
    if group_by == "l":
        return f"atom{col.atom + 1}:{col.label}"
    ml = _M_LABELS.get(col.l, [str(m) for m in range(2 * col.l + 1)])
    suffix = ml[col.m] if col.m < len(ml) else str(col.m)
    return f"atom{col.atom + 1}:{col.label}{('_' + suffix) if suffix else ''}"


def _unpack_result(res):
    """(system, nspin, eig[None-padded to (nspin,...)], coeffs[spin][k], fermi,
    device, weight_fn) for a norm-conserving SCFResult or a USPPResult."""
    formalism = getattr(res, "formalism", None)
    if isinstance(res, SCFResult):
        system = res.system
        nspin = int(getattr(res, "nspin", 1))
        eig = res.eigenvalues if nspin == 2 else res.eigenvalues[None]
        coeffs = res.coeffs if nspin == 2 else [res.coeffs]
        return (system, nspin, eig, coeffs, res.fermi, res.rho.device,
                _nc_weights_k)
    if formalism == "uspp":
        system = res.system
        nspin = int(res.nspin)
        eig = res.eigenvalues if nspin == 2 else res.eigenvalues[None]
        coeffs = res.coeffs if nspin == 2 else [res.coeffs]
        return (system, nspin, eig, coeffs, res.fermi, res.rho.device,
                _uspp_weights_k)
    raise NotImplementedError(
        "projected DOS supports the norm-conserving SCFResult and the USPP/PAW "
        "USPPResult; noncollinear is a separate path")


@torch.no_grad()
def projected_dos(res, *, width: float = 0.1, npoints: int = 800, window=None,
                  group_by: str = "l") -> ProjectedDOS:
    """Löwdin-projected DOS of a converged norm-conserving or USPP/PAW SCF.

    group_by is one of 'atom', 'l' (atom + orbital), 'lm' (adds m), or 'total'.
    Spin channels come back stacked on axis 0 for nspin=2.
    """
    system, nspin, eig, coeffs, fermi, device, weight_k = _unpack_result(res)
    cols = _atomic_columns(system)

    kw = system.kweights.to(device)
    g_spin = 2.0 if nspin == 1 else 1.0

    # per (spin, k) Löwdin weights and eigenvalues
    all_e = eig.reshape(nspin, -1).cpu().numpy()           # (nspin, nk*nb)
    weights = np.zeros((nspin, all_e.shape[1], len(cols)))  # (nspin, states, nproj)
    kweight_state = np.zeros((nspin, all_e.shape[1]))
    nb = eig.shape[-1]
    for isp in range(nspin):
        for ik, sph in enumerate(system.spheres):
            c = coeffs[isp][ik].to(device)                 # (nb, npw)
            wgt = weight_k(system, sph, ik, c, cols, device)  # (nb, nproj)
            sl = slice(ik * nb, (ik + 1) * nb)
            weights[isp, sl] = wgt
            kweight_state[isp, sl] = float(kw[ik])

    # spilling: 1 - <sum_p weight>_states, kweighted over every (spin, k, band).
    # A complete AO basis captures every state, so this is 0; the plane-wave
    # truncation and the finite orbital set leave a positive remainder.
    captured = (weights.sum(axis=2) * kweight_state).sum()
    spilling = float(1.0 - captured / kweight_state.sum())

    # energy grid + gaussian broadening
    window, grid = spectral_grid(all_e, width, npoints, window)

    def broaden(state_weight, isp):
        return _broaden(grid, all_e[isp],
                        kweight_state[isp] * g_spin * state_weight, width)

    labels = sorted({_group_key(c, group_by) for c in cols})
    groups = {}
    for lab in labels:
        mask = np.array([_group_key(c, group_by) == lab for c in cols])
        per_spin = [broaden(weights[isp][:, mask].sum(axis=1), isp)
                    for isp in range(nspin)]
        groups[lab] = per_spin[0] if nspin == 1 else np.stack(per_spin)
    total = [broaden(weights[isp].sum(axis=1), isp) for isp in range(nspin)]
    total = total[0] if nspin == 1 else np.stack(total)

    return ProjectedDOS(
        energy_eV=grid, total=total, groups=groups, spilling=spilling,
        fermi_eV=None if fermi is None else float(fermi),
        nspin=nspin, group_by=group_by,
    )


@torch.no_grad()
def projected_dos_noncollinear(res, *, width: float = 0.1, npoints: int = 800,
                               window=None, group_by: str = "l") -> NoncollinearPDOS:
    """Charge and spin-texture projected DOS of a noncollinear spinor SCF.

    Each spinor band is projected onto the pseudo-atomic orbitals per spin
    component, Löwdin-orthonormalized against the shared spatial overlap. The
    Pauli decomposition of the resulting amplitudes gives the charge n(E) and the
    spin-texture m_x/m_y/m_z(E), resolved by atom and orbital. The projection uses
    scalar AOs, so it applies with or without spin-orbit coupling (the j-resolved
    split is projected_dos_soc)."""
    from gradwave.scf.noncollinear import NCResult
    if not isinstance(res, NCResult):
        raise NotImplementedError(
            "projected_dos_noncollinear expects a noncollinear NCResult")
    system = res.system
    device = res.coeffs.device
    cols = _atomic_columns(system)
    m_pw = system.batch.npw_max
    kw = system.kweights.to(device)

    eig = res.eigenvalues                       # (nk, nb)
    nk, nb = eig.shape
    all_e = eig.reshape(-1).cpu().numpy()        # (nk*nb,)
    nstate = all_e.shape[0]
    n_pop = np.zeros((nstate, len(cols)))        # charge population per AO
    mx = np.zeros((nstate, len(cols)))
    my = np.zeros((nstate, len(cols)))
    mz = np.zeros((nstate, len(cols)))
    kweight_state = np.zeros(nstate)

    for ik, sph in enumerate(system.spheres):
        npw = sph.npw
        c = res.coeffs[ik].to(device)                       # (nb, 2·m_pw)
        cu, cd = split_spinor(c, npw, m_pw)                  # up / down components
        pu, pd = spinor_scalar_amplitudes(system, sph, cols, cu, cd, device)
        au, ad = (pu.real ** 2 + pu.imag ** 2), (pd.real ** 2 + pd.imag ** 2)
        cross = pu.conj() * pd                               # <phi~|up>* <phi~|down>
        sl = slice(ik * nb, (ik + 1) * nb)
        n_pop[sl] = (au + ad).cpu().numpy()
        mz[sl] = (au - ad).cpu().numpy()
        mx[sl] = (2.0 * cross.real).cpu().numpy()
        my[sl] = (2.0 * cross.imag).cpu().numpy()
        kweight_state[sl] = float(kw[ik])

    # spilling on the charge channel (each spinor band holds one electron, g=1)
    captured = (n_pop.sum(axis=1) * kweight_state).sum()
    spilling = float(1.0 - captured / kweight_state.sum())

    window, grid = spectral_grid(all_e, width, npoints, window)

    def chan(pop, mask):
        return _broaden(grid, all_e, kweight_state * pop[:, mask].sum(axis=1), width)

    labels = sorted({_group_key(c, group_by) for c in cols})
    masks = {lab: np.array([_group_key(c, group_by) == lab for c in cols])
             for lab in labels}
    charge = {lab: chan(n_pop, msk) for lab, msk in masks.items()}
    m_x = {lab: chan(mx, msk) for lab, msk in masks.items()}
    m_y = {lab: chan(my, msk) for lab, msk in masks.items()}
    m_z = {lab: chan(mz, msk) for lab, msk in masks.items()}
    full = np.ones(len(cols), dtype=bool)
    total_charge = chan(n_pop, full)

    return NoncollinearPDOS(
        energy_eV=grid, total_charge=total_charge, charge=charge,
        m_x=m_x, m_y=m_y, m_z=m_z, spilling=spilling,
        fermi_eV=None if res.fermi is None else float(res.fermi),
        group_by=group_by,
    )


@dataclass
class SOColumn:
    """One spin-angular (j, mj) atomic-orbital projector column."""

    atom: int
    species: int
    label: str        # e.g. "6P"
    l: int
    j: float          # l ± 1/2
    mj: float         # -j .. j


def _atomic_columns_so(system) -> list[SOColumn]:
    """Every PP_PSWFC orbital expanded over (j, mj); j comes from the FR pseudo."""
    cols = []
    for a, sp in enumerate(system.species_of_atom):
        u = system.upfs[sp]
        orbs = getattr(u, "pswfc", ())
        if not orbs:
            raise ValueError(
                f"{u.element}: the pseudopotential carries no PP_PSWFC atomic "
                "orbitals, so a j-resolved PDOS is not available")
        for o in orbs:
            j = getattr(o, "j", None)
            if j is None:
                raise ValueError(
                    f"{u.element}: PP_PSWFC orbitals carry no total angular "
                    "momentum j, so a j-resolved PDOS needs a fully-relativistic "
                    "pseudo (use projected_dos_noncollinear for scalar orbitals)")
            for imj in range(int(round(2 * j + 1))):
                cols.append(SOColumn(a, sp, o.label, o.l, float(j), -j + imj))
    return cols


def _ao_spinor_projectors_k(system, sph, cols, device):
    """Spin-angular AO projectors on one G-sphere, returned as separate up/down
    components qu, qd (each nproj, npw). Each |l, j, mj> is the Clebsch-Gordan
    combination c_up Y_l^{mj-1/2} chi_up + c_dn Y_l^{mj+1/2} chi_dn, the same
    construction core/spinor_proj.py uses for the SOC beta projectors."""
    from gradwave.core.spinor_proj import _cg, complex_ylm
    vol = system.grid.volume
    kpg = sph.kpg.to(device)
    npw = sph.npw
    qmag = np.sqrt(sph.kpg2.cpu().numpy())
    lmax = max(c.l for c in cols)
    yc = complex_ylm(lmax, kpg)  # (npw, (lmax+1)^2), index l^2 + l + m
    # radial form factors F_nl(q), one per (species, label, j) since j splits R_nl
    fcache: dict[tuple, torch.Tensor] = {}
    for sp in set(system.species_of_atom):
        u = system.upfs[sp]
        for o in u.pswfc:
            key = (sp, o.label, float(o.j))
            if key not in fcache:
                fcache[key] = torch.as_tensor(
                    sbt(o.l, o.rchi * u.r, u.r, u.rab, qmag),
                    dtype=RDTYPE, device=device).to(CDTYPE)
    phase_arg = kpg @ system.positions.to(device).T
    phase = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

    qu = torch.zeros(len(cols), npw, dtype=CDTYPE, device=device)
    qd = torch.zeros(len(cols), npw, dtype=CDTYPE, device=device)
    for p, c in enumerate(cols):
        pref = (4.0 * math.pi / math.sqrt(vol)) * _MINUS_I_POW[c.l]
        base = pref * fcache[(c.species, c.label, c.j)] * phase[:, c.atom]
        c_up, m_up, c_dn, m_dn = _cg(c.l, c.j, c.mj)
        if m_up is not None:
            qu[p] = base * (c_up * yc[:, c.l * c.l + c.l + m_up])
        if m_dn is not None:
            qd[p] = base * (c_dn * yc[:, c.l * c.l + c.l + m_dn])
    return qu, qd


def _group_key_so(col: SOColumn, group_by: str):
    if group_by == "total":
        return "total"
    if group_by == "atom":
        return f"atom{col.atom + 1}"
    if group_by == "l":
        return f"atom{col.atom + 1}:{col.label}"
    jtag = f"j{col.j:.1f}"
    if group_by == "jmj":
        return f"atom{col.atom + 1}:{col.label}_{jtag}_mj{col.mj:+.1f}"
    return f"atom{col.atom + 1}:{col.label}_{jtag}"           # group_by == "j"


@torch.no_grad()
def projected_dos_soc(res, *, width: float = 0.1, npoints: int = 800, window=None,
                      group_by: str = "j") -> ProjectedDOS:
    """j-resolved projected DOS of a fully-relativistic spinor SCF.

    Projects the spinor states onto spin-angular atomic orbitals |n l j mj> built
    from the FR pseudo's PP_PSWFC radials and Clebsch-Gordan coefficients, so a
    spin-orbit-split shell (e.g. p_{1/2} vs p_{3/2}) resolves into its j channels.
    group_by is 'j' (atom + orbital + j), 'jmj' (adds mj), 'l', 'atom', or 'total'.
    """
    from gradwave.scf.noncollinear import NCResult
    if not isinstance(res, NCResult):
        raise NotImplementedError(
            "projected_dos_soc expects a fully-relativistic noncollinear NCResult")
    system = res.system
    if not getattr(system, "is_fr", False):
        raise NotImplementedError(
            "j-resolved PDOS needs a fully-relativistic (SOC) pseudo; use "
            "projected_dos_noncollinear for scalar-relativistic noncollinear SCF")
    device = res.coeffs.device
    cols = _atomic_columns_so(system)
    m_pw = system.batch.npw_max
    kw = system.kweights.to(device)

    eig = res.eigenvalues                       # (nk, nb)
    nk, nb = eig.shape
    all_e = eig.reshape(-1).cpu().numpy()
    nstate = all_e.shape[0]
    weights = np.zeros((nstate, len(cols)))
    kweight_state = np.zeros(nstate)
    for ik, sph in enumerate(system.spheres):
        npw = sph.npw
        c = res.coeffs[ik].to(device)                       # (nb, 2·m_pw)
        cu, cd = split_spinor(c, npw, m_pw)
        proj = spinor_jmj_amplitudes(system, sph, cols, cu, cd, device)
        wgt = (proj.real ** 2 + proj.imag ** 2).cpu().numpy()
        sl = slice(ik * nb, (ik + 1) * nb)
        weights[sl] = wgt
        kweight_state[sl] = float(kw[ik])

    captured = (weights.sum(axis=1) * kweight_state).sum()
    spilling = float(1.0 - captured / kweight_state.sum())
    window, grid = spectral_grid(all_e, width, npoints, window)

    def chan(mask):
        return _broaden(grid, all_e, kweight_state * weights[:, mask].sum(axis=1),
                        width)

    labels = sorted({_group_key_so(c, group_by) for c in cols})
    groups = {lab: chan(np.array([_group_key_so(c, group_by) == lab for c in cols]))
              for lab in labels}
    total = chan(np.ones(len(cols), dtype=bool))
    return ProjectedDOS(
        energy_eV=grid, total=total, groups=groups, spilling=spilling,
        fermi_eV=None if res.fermi is None else float(res.fermi),
        nspin=1, group_by=group_by,
    )
