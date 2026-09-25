"""Baseline: the shipped BatchedHamiltonian.apply verbatim (FFT local path,
fp64). The population is scored relative to this."""


def apply(H, c):
    return H.apply(c)


__all__ = ["apply"]
