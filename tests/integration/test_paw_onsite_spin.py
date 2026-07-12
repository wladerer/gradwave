"""One-center spin degenerate limit: [ρ/2, ρ/2] must reproduce the
unpolarized E_1c AND ddd to machine precision.

The energy-only ζ=0 checks in tests/unit/test_spin_xc.py cannot see the GGA
vector field h_σ (the energy never uses it); ddd can. A missing chain-rule
factor on the ∂e/∂σ_tot cross term shifted ddd by 2% while E_1c stayed
exact — caught by the spin-Si degenerate-limit force/stress comparison.
"""

from pathlib import Path

import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.dtypes import CDTYPE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.paw_onsite import OneCenter

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


def test_onecenter_spin_degenerate_limit():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    nm = sum(2 * b.l + 1 for b in paw.betas)
    m0 = torch.zeros(nm, nm, dtype=CDTYPE)
    col = 0
    for i, b in enumerate(paw.betas):
        for _m in range(2 * b.l + 1):
            m0[col, col] = paw.paw_occ[i] / (2 * b.l + 1)
            col += 1
    # break sphericity so every Gaunt channel (and thus every h_lm) is live
    gen = torch.Generator().manual_seed(7)
    p = 0.03 * torch.randn(nm, nm, generator=gen, dtype=torch.float64)
    m0 = m0 + ((p + p.T) / 2).to(CDTYPE)

    e1, ddd1 = OneCenter(paw, PBE()).energy_and_ddd(m0)
    e2, ddd2 = OneCenter(paw, SpinPBE()).energy_and_ddd([m0 / 2, m0 / 2])

    assert abs(float(e1) - float(e2)) < 1e-10
    for d in ddd2:
        assert float((d - ddd1).abs().max()) < 1e-12
