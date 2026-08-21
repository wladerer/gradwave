"""Lean TR-validity test: sigma_shielding_dq on TR-folded (nk=36) vs unfolded
(nk=64) Si 4^3. If the two sigma agree, the analytic route already banks TR and
the incremental factor is post-TR (2.77x @4^3). If they differ, the route is
only correct unfolded and the incremental factor reverts to the raw 3.20x."""
import time, numpy as np, torch
from gradwave.constants import RY_EV
from gradwave.scf.loop import setup_system, scf
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq
RY=RY_EV; torch.manual_seed(0)
PSEUDO="tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf"
a=5.43; cell=a/2*np.array([[0.,1,1],[1,0,1],[1,1,0]]); pos=np.array([[0.,0,0],[a/4]*3])
def run(tr):
    sysd=setup_system(cell,pos,[0,0],[parse_upf(PSEUDO)],ecut=12*RY,kmesh=(4,4,4),
                      nbands=8,use_symmetry=False,time_reversal=tr,fft_shape=(20,20,20))
    res=scf(sysd,PBE(),etol=1e-9,rhotol=1e-8,verbose=False,max_iter=120)
    nk=len(sysd.spheres); t=time.perf_counter(); s=sigma_shielding_dq(res); w=time.perf_counter()-t
    print(f"# tr={tr} nk={nk} wall={w:.1f}s sigma_iso[Si0]={float(torch.diagonal(s[0]).mean()):+.3f}ppm",flush=True)
    return nk,s
nkt,st=run(True)
nkf,sf=run(False)
print(f"# |dq(TR nk={nkt}) - dq(unfold nk={nkf})|_max = {float((st-sf).abs().max()):.3e} ppm")
print(f"# (small => analytic route TR-valid => incremental=post-TR 2.77x; "
      f"large => route needs unfolded => incremental=raw 3.20x)")
print("# EXIT=0")
