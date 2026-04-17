"""
Synthetic Dataset Generator
============================
Simulates a multimodal neurorehabilitation dataset:
  - DTI (diffusion tensor imaging) features  → corticospinal tract integrity
  - EMG features                              → spinal/peripheral excitability
  - Kinematics features                       → motor performance
  - Clinical features                         → patient profile
  - Target: responder to neuromodulation (>20% improvement in motor score)

Based on patterns from published SCI/stroke neuromodulation literature.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_dataset(n=200, save_path=None):
    # ── Patient demographics ────────────────────────────────────────────────
    age          = np.random.normal(52, 14, n).clip(18, 80)
    injury_type  = np.random.choice(['SCI_incomplete', 'SCI_complete', 'stroke'], n,
                                     p=[0.40, 0.25, 0.35])
    months_post  = np.random.exponential(18, n).clip(1, 120)   # months since injury
    baseline_mss = np.random.normal(35, 12, n).clip(5, 60)     # baseline motor score (0–100)

    # ── DTI features (corticospinal tract integrity) ─────────────────────────
    # FA: fractional anisotropy  (0–1, higher = better tract integrity)
    # MD: mean diffusivity       (lower = better)
    # CST asymmetry index        (0 = symmetric, 1 = fully asymmetric)
    fa_cst       = np.random.beta(4, 2, n) * 0.7 + 0.1          # 0.1–0.8
    md_cst       = np.random.normal(0.85, 0.12, n).clip(0.5, 1.3)
    cst_asymmetry= np.random.beta(2, 5, n)                       # skewed low (most patients have some)

    # ── EMG features (neurophysiology) ──────────────────────────────────────
    # MEP amplitude (mV) — motor evoked potential via TMS
    # H-reflex ratio    — spinal excitability index
    # Voluntary EMG RMS — residual voluntary muscle activity
    mep_amplitude = np.random.lognormal(0.5, 0.8, n).clip(0.01, 8.0)
    h_reflex_ratio= np.random.beta(3, 3, n)
    emg_rms_vol   = np.random.lognormal(-0.2, 0.7, n).clip(0.01, 5.0)

    # ── Kinematics features (biomechanics) ──────────────────────────────────
    # Gait speed (m/s)     — for SCI/stroke; 0 = non-ambulatory
    # Step symmetry index  — 0=perfect, 1=fully asymmetric
    # Range of motion (°)  — upper/lower limb
    # Reaction time (ms)
    gait_speed    = np.random.gamma(2, 0.3, n).clip(0, 1.8)
    step_symmetry = np.random.beta(2, 4, n)
    rom_degrees   = np.random.normal(85, 25, n).clip(10, 160)
    reaction_time = np.random.normal(350, 80, n).clip(150, 700)

    # ── Generate response label (biologically plausible) ────────────────────
    # Responders tend to have: higher FA, lower asymmetry, higher MEP, younger age,
    # shorter time post-injury, higher baseline score, incomplete SCI or stroke
    logit = (
        2.5  * fa_cst                          # strong DTI predictor
       - 1.5 * cst_asymmetry                  # asymmetry → worse prognosis
        + 1.2 * mep_amplitude / 8.0            # MEP amplitude (normalised)
        + 0.8 * h_reflex_ratio                 # spinal excitability
        + 0.5 * emg_rms_vol / 5.0              # residual voluntary drive
        - 0.02 * age                           # age effect
        - 0.015 * months_post                  # time since injury
        + 0.02 * baseline_mss                  # baseline motor
        + 0.6  * (injury_type == 'SCI_incomplete').astype(float)
        + 0.4  * (injury_type == 'stroke').astype(float)
        - 1.0                                  # intercept (prevalence ~45%)
        + np.random.normal(0, 0.4, n)          # noise
    )
    prob_respond = 1 / (1 + np.exp(-logit))
    responder    = (np.random.uniform(size=n) < prob_respond).astype(int)

    df = pd.DataFrame({
        # Demographics
        'age':             age.round(1),
        'injury_type':     injury_type,
        'months_post_injury': months_post.round(1),
        'baseline_motor_score': baseline_mss.round(1),
        # DTI
        'dti_fa_cst':      fa_cst.round(4),
        'dti_md_cst':      md_cst.round(4),
        'dti_cst_asymmetry': cst_asymmetry.round(4),
        # EMG / Neurophysiology
        'mep_amplitude_mv':  mep_amplitude.round(3),
        'h_reflex_ratio':    h_reflex_ratio.round(3),
        'emg_rms_voluntary': emg_rms_vol.round(3),
        # Kinematics
        'gait_speed_ms':   gait_speed.round(3),
        'step_symmetry':   step_symmetry.round(3),
        'rom_degrees':     rom_degrees.round(1),
        'reaction_time_ms': reaction_time.round(1),
        # Target
        'responder':       responder,
    })

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved → {save_path}  ({n} patients, {responder.sum()} responders)")

    return df


if __name__ == '__main__':
    df = generate_dataset(n=200, save_path='data/neuro_multimodal.csv')
    print(df.describe().T[['mean','std','min','max']].round(3))
