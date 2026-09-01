"""task: optics — independent-particle / RPA dielectric ε(ω) on silicon.

A tiny NC Si SCF, then the interband ε₂(ω) + Kramers-Kronig ε₁ + absorption:
absorption is non-negative, vanishes below the gap (ε₂(0)≈0), and the static
ε₁(0) > 1. Also checks the full task wiring via api.run.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.optics import optical_epsilon
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY, si_fcc, si_upf

pytestmark = pytest.mark.standard

CELL, POS = si_fcc()


def _si_scf():
    upf = si_upf()
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY,
                          kmesh=(4, 4, 4), nbands=8)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-8, rhotol=1e-7,
              verbose=False)
    assert res.converged
    return res


def _si_system():
    return setup_system(CELL, POS, [0, 0], [si_upf()], ecut=15 * RY,
                        kmesh=(4, 4, 4), nbands=8)


def test_optics_epsilon_si():
    torch.set_num_threads(8)
    res = _si_scf()
    om, eps1, eps2, alpha, info = optical_epsilon(
        res, omega_max=20.0, n_omega=400, eta=0.1, n_extra_bands=8, verbose=False)

    assert om.shape == eps1.shape == eps2.shape == alpha.shape == (400,)
    assert np.all(np.isfinite(eps2)) and (eps2 >= -1e-8).all()  # absorption ≥ 0
    assert np.all(alpha >= -1e-6)
    assert eps2[0] < 0.05 * eps2.max()  # gapped: only Lorentzian tail at ω→0
    assert 5.0 < info["eps_static"] < 40.0  # LDA Si ε₁(0) ≈ 15 (IP, no local fields)
    assert info["n_occ"] == 4            # Si: 8 valence e⁻ → 4 occupied bands
    # absorption onset sits above the (LDA-underestimated) gap
    assert om[np.argmax(eps2 > 1.0)] > 1.0
    # anisotropic tensor: cubic Si is isotropic → xx≈yy≈zz, and their mean is ε₂
    exx, eyy, ezz = (np.array(info["eps2_tensor"][i]) for i in range(3))
    assert np.allclose(exx, eyy, atol=0.2) and np.allclose(exx, ezz, atol=0.2)
    assert np.allclose((exx + eyy + ezz) / 3.0, eps2, atol=1e-9)
    assert info["velocity"] == "full"


def test_optics_velocity_and_local_fields():
    """velocity='local' runs; local-field effects reduce ε₁(0) (RPA screening)."""
    res = _si_scf()
    _, e1_full, _, _, _ = optical_epsilon(res, omega_max=16.0, n_omega=200, eta=0.15,
                                          n_extra_bands=6, velocity="full")
    _, e1_loc, _, _, info_loc = optical_epsilon(res, omega_max=16.0, n_omega=200,
                                                eta=0.15, n_extra_bands=6,
                                                velocity="local")
    assert info_loc["velocity"] == "local"
    assert 5.0 < info_loc["eps_static"] < 40.0
    # nonlocal commutator shifts ε₁(0) but keeps it physical
    assert abs(e1_full[0] - e1_loc[0]) < 10.0

    _, e1_lfe, e2_lfe, _, info_lfe = optical_epsilon(
        res, omega_max=16.0, n_omega=120, eta=0.2, n_extra_bands=6, local_fields=True)
    assert info_lfe["local_fields"] is True
    assert np.all(np.isfinite(e2_lfe))
    # local fields screen the response DOWN vs the same-convention no-LFE head
    assert info_lfe["eps_static"] > 1.0
    assert info_lfe["eps_static"] <= info_lfe["eps1_nolfe"][0] * 1.02


def test_optics_scissor():
    """A scissor shift blue-shifts the ε₂ peak by ~Δ and lowers ε₁(0)."""
    res = _si_scf()
    om, _, e2_0, _, info0 = optical_epsilon(
        res, omega_max=16.0, n_omega=320, eta=0.15, n_extra_bands=6, scissor=0.0)
    _, _, e2_s, _, info_s = optical_epsilon(
        res, omega_max=16.0, n_omega=320, eta=0.15, n_extra_bands=6, scissor=1.0)
    peak0, peak_s = om[e2_0.argmax()], om[e2_s.argmax()]
    assert peak_s - peak0 > 0.5              # peak moves up
    assert abs((peak_s - peak0) - 1.0) < 0.4  # by ~ the scissor
    # oscillator strengths preserved, gap opened → static ε₁(0) drops
    assert info_s["eps_static"] < info0["eps_static"]
    assert info_s["scissor_eV"] == 1.0


def test_optics_nspin2_matches_spin_restricted():
    """Collinear nspin=2 on non-magnetic Si (start_mag=0) reproduces the nspin=1
    spectrum — the two-channel sum and the halved spin fold cancel exactly."""
    torch.set_num_threads(8)
    r1 = _si_scf()
    # small gaussian smearing (the spinor/collinear-spin drivers require a smearing
    # scheme); Si's gap ≫ width so occupations stay integer and the comparison holds
    r2 = scf(_si_system(), LSDA_PW92(), smearing="gaussian", width=0.05, nspin=2,
             start_mag=[0.0, 0.0], etol=1e-8, rhotol=1e-7, verbose=False)
    assert r2.converged
    kw = dict(omega_max=16.0, n_omega=240, eta=0.15, n_extra_bands=6)
    _, e1_1, e2_1, _, info1 = optical_epsilon(r1, **kw)
    _, e1_2, e2_2, _, info2 = optical_epsilon(r2, **kw)
    assert info2["nspin"] == 2 and info1["nspin"] == 1
    assert info2["n_occ"] == 4                       # per-channel occupied count
    # non-magnetic: the two spin channels are identical to the spin-paired result
    assert abs(info2["eps_static"] - info1["eps_static"]) < 0.2
    assert np.abs(e2_2 - e2_1).max() < 0.3 * e2_1.max()
    assert np.allclose(e1_2, e1_1, atol=0.4)


def test_optics_noncollinear_matches_spin_restricted():
    """Noncollinear/spinor optics on non-magnetic Si (m⃗≡0 seed) reduces to the
    nspin=1 spectrum: the spinor path (rebuilt potential, doubled bands, g=1) is
    the same physics with the spin fold bookkept."""
    torch.set_num_threads(8)
    r1 = _si_scf()
    xc = NoncollinearXC(LSDA_PW92())
    rnc = scf_noncollinear(_si_system(), xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.05, etol=1e-8, rhotol=1e-7,
                           verbose=False)
    assert rnc.converged
    assert float(rnc.m.abs().max()) < 1e-6          # stayed non-magnetic
    kw = dict(omega_max=16.0, n_omega=240, eta=0.15)
    _, e1_1, e2_1, _, info1 = optical_epsilon(r1, n_extra_bands=6, **kw)
    # spinor bands are Kramers-doubled, so 2× the conduction bands cover the same
    # spatial-orbital manifold as the nspin=1 reference (a fair truncation match)
    _, e1_n, e2_n, _, info_n = optical_epsilon(rnc, xc=xc, n_extra_bands=12, **kw)
    assert info_n["formalism"] == "noncollinear"
    assert info_n["n_occ"] == 8                       # 8 filled spinor bands (Si)
    assert abs(info_n["eps_static"] - info1["eps_static"]) < 0.3
    assert np.abs(e2_n - e2_1).max() < 0.35 * e2_1.max()
    assert np.allclose(e1_n, e1_1, atol=0.5)


def test_optics_noncollinear_needs_xc():
    """The spinor path errors clearly without the XC functional (NCResult carries
    no v_eff to rebuild the potential from)."""
    xc = NoncollinearXC(LSDA_PW92())
    rnc = scf_noncollinear(_si_system(), xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.05, etol=1e-8, rhotol=1e-7,
                           verbose=False)
    with pytest.raises(ValueError, match="needs the XC functional"):
        optical_epsilon(rnc, omega_max=12.0, n_omega=80, n_extra_bands=4)
    with pytest.raises(NotImplementedError, match="local fields.*noncollinear"):
        optical_epsilon(rnc, xc=xc, local_fields=True, omega_max=12.0, n_omega=80,
                        n_extra_bands=4)


def test_optics_params_and_output():
    """OpticsParams validation + the `.out` renderer."""
    from gradwave.inputs.models import InputError, OpticsParams
    from gradwave.io.output import _optics_lines

    for bad in (dict(eta=0.0), dict(n_omega=1), dict(omega_max=-1.0),
                dict(n_extra_bands=0)):
        with pytest.raises(InputError):
            OpticsParams(**bad)

    optics = {"omega_eV": [0.0, 10.0], "eps1": [12.0, 1.0], "eps2": [0.0, 5.0],
              "absorption_inv_cm": [0.0, 1e5], "eta_eV": 0.1, "n_bands": 12,
              "n_occ": 4, "eps_static": 12.0}
    text = "\n".join(_optics_lines(optics))
    assert "optical dielectric" in text
    assert "ε₁(0)" in text
    assert "4 occ + 8 cond" in text
