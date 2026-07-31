"""Phase 2: transverse-pin discriminator.

The socfree trace shows the transverse m channels growing from machine zero at
~3x per iteration to a 1e-4 floor: a genuine instability of the mixing map on
the soft (magnon) transverse modes. Pin them (zero the m_x/m_y blocks of the
mixing vector each iteration, exact for a collinear-limit state) and see which
arms then converge and hold the moment:

  pin_pulay_quad       socfree, stock pulay + quadratic schedule + pin
  pin_johnson_quad     socfree, johnson + quadratic + pin (does the collapse
                       seen without the pin persist?)
  johnson_quad_soft    socfree, johnson + quadratic, mag_mixing_alpha=0.3,
                       no pin (is the collapse a step-size artifact?)
  pin_soc_johnson_quad SOC Ni, johnson + quadratic + pin (how much transverse
                       does SOC genuinely regrow per iteration?)
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from experiments.noncollinear_convergence import systems as sysmod
from experiments.noncollinear_convergence.probe import NCConvergenceProbe
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.scf.noncollinear import scf_noncollinear

torch.set_num_threads(8)
outdir = Path("experiments/noncollinear_convergence/results") / "asus"

TOL = dict(smearing="gaussian", width=0.1, max_iter=80, etol=1e-6, rhotol=1e-5,
           diago_tol=1e-9, verbose=False)


class PinnedProbe(NCConvergenceProbe):
    """Record the raw residual, then zero the transverse (m_x, m_y) blocks of
    BOTH mixing vectors in place, so the mixed state never carries them."""

    def __init__(self, system, pin: bool):
        super().__init__(system, nonmagnetic=False)
        self.pin = pin

    def __call__(self, it, vin, vout):
        super().__call__(it, vin, vout)  # record regrowth before pinning
        if self.pin:
            ng = self.ng
            vin[ng:3 * ng] = 0.0
            vout[ng:3 * ng] = 0.0


def run(name, soc, pin, **kw):
    system = sysmod.ni_fcc(soc)
    probe = PinnedProbe(system, pin=pin)
    opts = dict(TOL)
    opts.update(kw)
    res = scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                           mag_vec_init=[[0.0, 0.0, 0.6]], mixer_hook=probe,
                           **opts)
    with open(outdir / f"trace_{name}.jsonl", "w") as f:
        for rec in probe.records:
            f.write(json.dumps(rec) + "\n")
    row = dict(name=name, converged=bool(res.converged), n_iter=int(res.n_iter),
               free_energy=float(res.energies.free_energy),
               mag_vec=[round(float(x), 4) for x in res.mag_vec],
               mag_abs=round(float(res.mag_abs), 4), pin=pin, soc=soc,
               opts={k: repr(v) for k, v in kw.items()})
    print(json.dumps(row), flush=True)
    with open(outdir / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


run("ni_pin_pulay_quad_s0.6_z", False, True, mag_diago_schedule="quadratic")
run("ni_pin_johnson_quad_s0.6_z", False, True, mag_mixer="johnson",
    mag_diago_schedule="quadratic")
run("ni_johnson_quad_soft_s0.6_z", False, False, mag_mixer="johnson",
    mag_diago_schedule="quadratic", mag_mixing_alpha=0.3)
run("ni_pin_soc_johnson_quad_s0.6_z", True, True, mag_mixer="johnson",
    mag_diago_schedule="quadratic")
print("DONE_PROTO2", flush=True)
