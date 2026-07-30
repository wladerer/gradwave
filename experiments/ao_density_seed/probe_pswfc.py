"""Probe which pseudos carry PSWFC orbitals and their l/occupation."""
import sys
from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw

def show(path, paw=False):
    try:
        d = parse_upf_paw(path) if paw else parse_upf(path)
    except Exception as e:
        print(f"  {path}: PARSE FAIL {type(e).__name__}: {e}")
        return
    orbs = d.chi if paw else d.pswfc
    zval = d.z_valence
    tag = "PAW" if paw else "NC"
    print(f"  [{tag}] {path.split('/')[-1]}  Zval={zval}  norb={len(orbs)}  nlcc={d.core_rho is not None}")
    for o in orbs:
        print(f"      l={o.l} label={o.label!r} occ={o.occupation}")

base = "tests/fixtures/qe/pseudos/"
print("NONMAG NC:")
for f in ["Si_ONCV_PBE-1.2.upf","Al_ONCV_PBE-1.2.upf","Mg_ONCV_PBE-1.2.upf","O_ONCV_PBE-1.2.upf","Na_ONCV_PBE_sr.upf","Cl_ONCV_PBE_sr.upf"]:
    show(base+f)
print("MAG NC:")
for f in ["Fe_ONCV_PBE-1.2.upf","PD_Ni_PBE.upf"]:
    show(base+f)
print("MAG PAW:")
for f in ["Fe.pbe-spn-kjpaw_psl.1.0.0.UPF","Ni.pbe-spn-kjpaw_psl.1.0.0.UPF"]:
    show(base+f, paw=True)
print("NONMAG PAW:")
for f in ["Si.pbe-n-kjpaw_psl.1.0.0.UPF","Al.pbe-n-kjpaw_psl.1.0.0.UPF"]:
    show(base+f, paw=True)
