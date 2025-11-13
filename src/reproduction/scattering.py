"""Scattering spectrum computation from Morlet wavelets."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .wavelets import MorletWaveletBank


@dataclass
class ScatteringSpectrum:
    """Compute low-order scattering statistics for time series.

    The implementation follows the general principles of wavelet scattering as
    introduced by Mallat and Bruna. For a discrete signal ``x[t]`` and a wavelet
    filter bank :class:`MorletWaveletBank`, the first layer computes complex
    wavelet responses ``Wx[j, t]``. Their modulus is averaged over time to
    produce first-order scattering coefficients. Higher-order interactions are
    obtained by convolving the modulus with additional wavelets.

    The phase/modulus cross-spectrum summarises the covariance between the
    complex wavelet coefficients and modulus responses at pairs of scales. It is
    particularly sensitive to non-Gaussian effects such as intermittency.
    """

    bank: MorletWaveletBank
    eps: float = 1e-8

    def analyse(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        responses = self.bank.convolve(signal)
        filters = self.bank.filters
        moduli = [np.abs(r) for r in responses]
        centred_moduli = [m - m.mean() for m in moduli]

        # First-order scattering: average modulus per scale.
        s1 = np.array([m.mean() for m in moduli])

        # Second-order scattering: cascade modulus through higher scale wavelets.
        s2_entries: List[float] = []
        s2_index: List[Tuple[int, int]] = []
        fft_moduli = [np.fft.rfft(m) for m in centred_moduli]
        for j1, j2 in itertools.permutations(range(len(responses)), 2):
            if self.bank.scales[j2] <= self.bank.scales[j1]:
                continue
            cascaded = np.fft.irfft(fft_moduli[j1] * filters[j2], n=self.bank.n_samples)
            s2_entries.append(np.mean(np.abs(cascaded)))
            s2_index.append((self.bank.scales[j1], self.bank.scales[j2]))
        s2 = np.array(s2_entries)

        # Phase-modulus covariance across scales.
        cross_entries: List[float] = []
        cross_index: List[Tuple[int, int]] = []
        for j1, j2 in itertools.permutations(range(len(responses)), 2):
            if self.bank.scales[j2] < self.bank.scales[j1]:
                continue
            r = responses[j1] - responses[j1].mean()
            cov = np.mean(r * centred_moduli[j2])
            cross_entries.append(np.abs(cov))
            cross_index.append((self.bank.scales[j1], self.bank.scales[j2]))
        cross = np.array(cross_entries)

        return {
            "S1": s1,
            "S2": s2,
            "S2_index": np.array(s2_index),
            "phase_modulus": cross,
            "phase_modulus_index": np.array(cross_index),
        }


__all__ = ["ScatteringSpectrum"]
