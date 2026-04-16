"""
MemoryShield hyperparameters and defaults.
"""
from dataclasses import dataclass


@dataclass
class MemShieldConfig:
    # ── FIFO Resonance Scheduler ─────────────────────────────────────────────
    fifo_window: int = 7          # SAM2 num_maskmem (FIFO bank capacity)
    resonance_period: int = 6     # Weak frame interval = fifo_window - 1
    max_insertion_ratio: float = 0.15

    # ── Perturbation budget ──────────────────────────────────────────────────
    epsilon_strong: float = 8.0 / 255
    epsilon_weak_ratio: float = 0.25  # Weak = ratio * strong budget
    epsilon_weak_floor: float = 2.0 / 255

    # ── PGD optimization ─────────────────────────────────────────────────────
    n_steps_strong: int = 300
    n_steps_weak: int = 100
    lr: float = 2e-3

    # ── Quality constraints ──────────────────────────────────────────────────
    ssim_threshold_strong: float = 0.93
    ssim_threshold_weak: float = 0.97
    lambda_quality: float = 5.0

    # ── Loss ─────────────────────────────────────────────────────────────────
    future_horizon: int = 15      # Evaluate attack on this many future frames
    persistence_weighting: bool = True

    # ── Content analyzer ─────────────────────────────────────────────────────
    enable_occlusion_ghost: bool = True
    enable_topology_seed: bool = True
    occlusion_flow_threshold: float = 0.3
    topology_narrow_px: int = 8

    # ── Codec robustness ─────────────────────────────────────────────────────
    codec_in_loop: bool = False

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = "cuda"

    @property
    def epsilon_weak(self) -> float:
        return max(self.epsilon_strong * self.epsilon_weak_ratio,
                   self.epsilon_weak_floor)
