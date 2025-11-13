"""Complete experimental pipeline used for the reproduction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .processes import (
    simulate_brownian_motion,
    simulate_hawkes_process,
    simulate_mrw,
    simulate_poisson_process,
    simulate_sp500_surrogate,
    simulate_turbulence_surrogate,
)
from .scattering import ScatteringSpectrum
from .wavelets import MorletWaveletBank


RESULTS_DIR = Path("results")


@dataclass
class ExperimentConfig:
    n_samples: int = 4096
    scales: Tuple[int, ...] = tuple(range(1, 8))
    central_freq: float = 6.0
    bandwidth: float = 1.0
    seed: int = 1234


def build_bank(cfg: ExperimentConfig) -> MorletWaveletBank:
    return MorletWaveletBank(
        n_samples=cfg.n_samples,
        scales=cfg.scales,
        central_freq=cfg.central_freq,
        bandwidth=cfg.bandwidth,
    )


def generate_signals(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    seeds = rng.integers(0, 2**31 - 1, size=6)
    return {
        "brownian": simulate_brownian_motion(cfg.n_samples, seed=int(seeds[0])),
        "poisson": simulate_poisson_process(cfg.n_samples, rate=4.0, seed=int(seeds[1])),
        "mrw": simulate_mrw(cfg.n_samples, seed=int(seeds[2])),
        "hawkes": simulate_hawkes_process(cfg.n_samples, seed=int(seeds[3])),
        "turbulence": simulate_turbulence_surrogate(cfg.n_samples, seed=int(seeds[4])),
        "sp500": simulate_sp500_surrogate(cfg.n_samples, seed=int(seeds[5])),
    }


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Dict[str, np.ndarray]]:
    bank = build_bank(cfg)
    analyser = ScatteringSpectrum(bank)
    signals = generate_signals(cfg)
    summaries: Dict[str, Dict[str, np.ndarray]] = {}
    for name, sig in signals.items():
        sig = sig - sig.mean()
        summaries[name] = analyser.analyse(sig)
    return summaries


def export_summary(summary: Dict[str, Dict[str, np.ndarray]], cfg: ExperimentConfig) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    # Save raw numerical arrays for reproducibility.
    serialisable = {
        name: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}
        for name, stats in summary.items()
    }
    (RESULTS_DIR / "scattering_summary.json").write_text(json.dumps(serialisable, indent=2))


def plot_spectra(summary: Dict[str, Dict[str, np.ndarray]], cfg: ExperimentConfig) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    scales = np.array(cfg.scales)

    # Plot first-order scattering magnitudes.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, stats in summary.items():
        ax.plot(scales, stats["S1"], marker="o", label=name)
    ax.set_xlabel("Wavelet scale j")
    ax.set_ylabel("First-order scattering magnitude")
    ax.set_title("First-order scattering spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "first_order_spectra.png", dpi=200)
    plt.close(fig)

    # Phase-modulus covariance heatmaps.
    for name, stats in summary.items():
        pm = stats["phase_modulus"]
        index = stats["phase_modulus_index"]
        fig, ax = plt.subplots(figsize=(5, 4))
        heatmap = np.zeros((len(scales), len(scales)))
        for (j1, j2), value in zip(index, pm):
            i = np.where(scales == j1)[0][0]
            k = np.where(scales == j2)[0][0]
            heatmap[i, k] = value
        im = ax.imshow(heatmap, origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(scales)))
        ax.set_xticklabels(scales)
        ax.set_yticks(np.arange(len(scales)))
        ax.set_yticklabels(scales)
        ax.set_xlabel("Modulus scale")
        ax.set_ylabel("Phase scale")
        ax.set_title(f"Phase-modulus covariance ({name})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Cov|")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"phase_modulus_{name}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    cfg = ExperimentConfig()
    summary = run_experiment(cfg)
    export_summary(summary, cfg)
    plot_spectra(summary, cfg)


if __name__ == "__main__":
    main()
