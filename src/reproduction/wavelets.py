"""Utilities to build complex Morlet wavelet filter banks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class MorletWaveletBank:
    """Create a Morlet wavelet filter bank for one-dimensional signals.

    Parameters
    ----------
    n_samples:
        Length of the discrete signal to be analysed.
    scales:
        Iterable of integer dyadic scales ``j`` defining wavelets with width
        ``2**j`` samples. The smallest meaningful scale should be large enough
        compared to the sampling period so that the wavelets remain well
        localised.
    central_freq:
        Central frequency multiplier :math:`\omega_0` of the Morlet wavelet.
        Larger values lead to more oscillations.
    bandwidth:
        Controls the spread of the Gaussian envelope. The default value of
        ``1.0`` corresponds to the canonical Morlet definition and works well
        for most experiments.
    """

    n_samples: int
    scales: Iterable[int]
    central_freq: float = 6.0
    bandwidth: float = 1.0

    def __post_init__(self) -> None:
        self.scales = list(sorted(self.scales))
        if self.n_samples <= 0:
            raise ValueError("n_samples must be strictly positive")
        if not self.scales:
            raise ValueError("At least one scale must be provided")
        self._fft_frequencies = 2 * math.pi * np.fft.rfftfreq(self.n_samples)
        self._filters = self._build_filters()

    def _build_filters(self) -> List[np.ndarray]:
        filters: List[np.ndarray] = []
        for scale in self.scales:
            width = float(2 ** scale)
            # Analytic Morlet wavelet in Fourier domain.
            # We rely on the approximation given by a Gaussian centred at
            # ``central_freq / width`` and cancel the zero-frequency component
            # so that the wavelet has a negligible mean.
            freq = self._fft_frequencies * width
            envelope = np.exp(-0.5 * (freq - self.central_freq) ** 2 / (self.bandwidth ** 2))
            # Enforce analyticity by zeroing negative frequencies
            # (their contribution is absent in the rfft representation).
            envelope[0] = 0.0
            # Normalise to unit energy in the time domain.
            normalisation = np.sqrt(np.sum(np.abs(envelope) ** 2))
            if normalisation == 0:
                raise ValueError("Degenerate wavelet filter encountered")
            filters.append(envelope / normalisation)
        return filters

    @property
    def filters(self) -> List[np.ndarray]:
        return list(self._filters)

    def convolve(self, signal: np.ndarray) -> List[np.ndarray]:
        """Compute wavelet convolutions for all scales.

        Parameters
        ----------
        signal:
            One-dimensional real valued signal of length ``n_samples``.

        Returns
        -------
        list of ``np.ndarray``
            Complex valued wavelet responses, one for each scale in
            ``self.scales``.
        """

        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if signal.shape[0] != self.n_samples:
            raise ValueError("signal length mismatch")

        spectrum = np.fft.rfft(signal)
        responses: List[np.ndarray] = []
        for filt in self._filters:
            conv = np.fft.irfft(spectrum * filt, n=self.n_samples)
            responses.append(conv)
        return responses


__all__ = ["MorletWaveletBank"]
