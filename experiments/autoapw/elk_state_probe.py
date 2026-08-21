"""Probe Elk STATE.OUT Fortran-unformatted record structure (read-only, tiny).

Reads sequential gfortran records (4-byte length markers) and prints each record's
byte length + a guess at its content, to locate the density (rhomt) and Coulomb
potential (vclmt) records per writestate.f90.
"""
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/wladerer/tio2_efg/STATE.OUT"


def records(fh):
    while True:
        head = fh.read(4)
        if len(head) < 4:
            return
        (n,) = struct.unpack("<i", head)
        data = fh.read(n)
        tail = fh.read(4)
        (n2,) = struct.unpack("<i", tail)
        assert n == n2, (n, n2)
        yield data


with open(path, "rb") as fh:
    for i, rec in enumerate(records(fh)):
        n = len(rec)
        guess = ""
        if n == 12:
            guess = f"3xint32={struct.unpack('<3i', rec)}"
        elif n == 4:
            guess = f"int32={struct.unpack('<i', rec)[0]} / logical"
        elif n == 8:
            guess = f"real8={struct.unpack('<d', rec)[0]:.6g} / 2xint32={struct.unpack('<2i', rec)}"
        elif n % 8 == 0 and n < 40000:
            guess = f"{n//8} real8 (first={struct.unpack('<d', rec[:8])[0]:.6g}, last={struct.unpack('<d', rec[-8:])[0]:.6g})"
        else:
            guess = f"BIG {n} bytes ({n/8:.0f} real8?)"
        print(f"rec {i:2d}: {n:>10d} bytes  {guess}")
        if i > 30:
            break
