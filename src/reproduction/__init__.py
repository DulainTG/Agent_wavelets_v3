"""Reproduction package for wavelet scattering experiments."""

from .wavelets import MorletWaveletBank
from .scattering import ScatteringSpectrum
from .processes import (
    simulate_brownian_motion,
    simulate_poisson_process,
    simulate_mrw,
    simulate_hawkes_process,
    simulate_turbulence_surrogate,
    simulate_sp500_surrogate,
)

__all__ = [
    "MorletWaveletBank",
    "ScatteringSpectrum",
    "simulate_brownian_motion",
    "simulate_poisson_process",
    "simulate_mrw",
    "simulate_hawkes_process",
    "simulate_turbulence_surrogate",
    "simulate_sp500_surrogate",
]
