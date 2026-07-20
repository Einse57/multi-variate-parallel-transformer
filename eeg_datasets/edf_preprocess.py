"""
EDF/EDF+ preprocessing to match the SWEC-ETHZ iEEG dataset conditioning.

The SWEC-ETHZ H5 files already have these steps applied; EDF recordings
from clinical systems typically do NOT.  This module applies the same
preprocessing pipeline so that EDF data is compatible with MVPFormer.

Reference (SWEC paper, Section 4):
  "The signals were median-referenced and digitally band-pass filtered
   between 0.5 and 120 Hz using a fourth-order Butterworth filter, both
   in a forward and backward pass to minimize phase distortions."

Pipeline steps (in order):
  1. Bandpass filter 0.5–120 Hz (4th-order Butterworth, zero-phase)
  2. Optional notch filter for powerline noise (50 or 60 Hz)
  3. Median re-reference (subtract cross-channel median per timepoint)
  4. Resample to target rate (512 Hz default)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def bandpass_butterworth(
    data: np.ndarray,
    sfreq: float,
    low: float = 0.5,
    high: float = 120.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : (channels, samples) array
    sfreq : Sampling frequency in Hz
    low : Low cutoff frequency (Hz)
    high : High cutoff frequency (Hz)
    order : Filter order

    Returns
    -------
    Filtered data, same shape as input.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = sfreq / 2.0
    # Clamp high to just below Nyquist
    high = min(high, nyq - 1.0)
    if low >= high:
        return data

    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(data.dtype)


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase notch (band-stop) filter for powerline removal.

    Parameters
    ----------
    data : (channels, samples) array
    sfreq : Sampling frequency in Hz
    freq : Notch frequency (50 Hz for EU, 60 Hz for US)
    quality : Quality factor (higher = narrower notch)

    Returns
    -------
    Filtered data, same shape as input.
    """
    from scipy.signal import iirnotch, sosfiltfilt

    nyq = sfreq / 2.0
    if freq >= nyq:
        return data

    b, a = iirnotch(freq / nyq, quality)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos

    sos = tf2sos(b, a)
    return sosfiltfilt(sos, data, axis=-1).astype(data.dtype)


def median_rereference(data: np.ndarray) -> np.ndarray:
    """Subtract the cross-channel median at each timepoint.

    Parameters
    ----------
    data : (channels, samples) array

    Returns
    -------
    Re-referenced data.
    """
    median = np.median(data, axis=0, keepdims=True)
    return data - median


def resample(
    data: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """Resample data to target sampling frequency.

    Uses scipy resample_poly (polyphase) when available for quality,
    falls back to linear interpolation.

    Parameters
    ----------
    data : (channels, samples) array
    orig_sfreq : Original sampling frequency
    target_sfreq : Desired sampling frequency

    Returns
    -------
    Resampled data.
    """
    if abs(orig_sfreq - target_sfreq) < 0.5:
        return data

    try:
        from math import gcd
        from scipy.signal import resample_poly

        up = int(target_sfreq)
        down = int(orig_sfreq)
        g = gcd(up, down)
        up, down = up // g, down // g
        return resample_poly(data, up, down, axis=-1).astype(data.dtype)
    except Exception:
        # Fallback: linear interpolation
        n_ch, n_samples = data.shape
        duration = n_samples / orig_sfreq
        new_n = int(round(duration * target_sfreq))
        old_times = np.linspace(0, duration, n_samples, endpoint=False)
        new_times = np.linspace(0, duration, new_n, endpoint=False)
        resampled = np.empty((n_ch, new_n), dtype=data.dtype)
        for ch in range(n_ch):
            resampled[ch] = np.interp(new_times, old_times, data[ch])
        return resampled


def detect_preprocessing(data: np.ndarray, sfreq: float) -> dict:
    """Heuristically detect whether data has already been preprocessed.

    Checks for the three hallmarks of the SWEC paper pipeline:
      1. Bandpass filtered (no DC offset, no significant sub-0.5 Hz energy)
      2. Median re-referenced (cross-channel median ≈ 0 at each timepoint)

    Returns a dict of booleans:
      - ``bandpass_needed``: True if bandpass appears NOT already applied
      - ``median_ref_needed``: True if median re-ref appears NOT already applied
    """
    # ── DC / bandpass check ──
    # A bandpass-filtered signal at 0.5 Hz should have mean ≈ 0.
    # Raw iEEG often has mean offsets of tens to hundreds of µV.
    ch_means = np.abs(data.mean(axis=-1))  # per-channel absolute mean
    median_abs_mean = float(np.median(ch_means))
    # Threshold: if median |mean| > 1 µV (in whatever unit the data uses),
    # there is likely DC content.  SWEC data is in µV with means < 0.01.
    # Use a relative check too: compare mean to std.
    ch_stds = data.std(axis=-1)
    median_std = float(np.median(ch_stds))
    # If mean is > 1% of std, there's meaningful DC that a bandpass would remove.
    dc_ratio = median_abs_mean / (median_std + 1e-12)
    bandpass_needed = dc_ratio > 0.01

    # ── Median re-reference check ──
    # After median re-referencing, the cross-channel median at each
    # timepoint should be ≈ 0.  Sample a subset to keep it fast.
    n_samples = data.shape[1]
    step = max(1, n_samples // 2000)  # check ~2000 timepoints
    subset = data[:, ::step]
    cross_ch_median = np.median(subset, axis=0)  # (timepoints,)
    median_of_medians = float(np.median(np.abs(cross_ch_median)))
    # Compare to channel amplitude
    median_ref_needed = (median_of_medians / (median_std + 1e-12)) > 0.01

    return {
        "bandpass_needed": bandpass_needed,
        "median_ref_needed": median_ref_needed,
        "_dc_ratio": dc_ratio,
        "_median_ratio": median_of_medians / (median_std + 1e-12),
    }


def preprocess_edf_for_mvpformer(
    data: np.ndarray,
    sfreq: float,
    *,
    target_sfreq: float = 512.0,
    bandpass: bool = True,
    bandpass_low: float = 0.5,
    bandpass_high: float = 120.0,
    bandpass_order: int = 4,
    notch: Optional[float] = None,
    notch_quality: float = 30.0,
    do_median_ref: bool = True,
    do_resample: bool = True,
    auto_detect: bool = False,
) -> tuple[np.ndarray, float]:
    """Full SWEC-paper preprocessing pipeline for raw EDF data.

    Parameters
    ----------
    data : (channels, samples) float32 array — raw EDF signal
    sfreq : Original sampling frequency of the EDF
    target_sfreq : Resample target (512 Hz matches MVPFormer training)
    bandpass : Whether to apply bandpass filter
    bandpass_low : Low cutoff (Hz)
    bandpass_high : High cutoff (Hz)
    bandpass_order : Butterworth filter order
    notch : Powerline notch frequency (None to skip, 50.0 or 60.0 typical)
    notch_quality : Q factor for notch filter
    do_median_ref : Whether to apply median re-referencing
    do_resample : Whether to resample to target_sfreq
    auto_detect : When True, run ``detect_preprocessing`` first and skip
        steps that appear already applied.  The explicit ``bandpass`` and
        ``do_median_ref`` flags are still honoured as overrides when
        ``auto_detect`` is False.

    Returns
    -------
    (processed_data, effective_sfreq) — the preprocessed array and its rate.
    """
    skipped: list[str] = []

    if auto_detect:
        det = detect_preprocessing(data, sfreq)
        if not det["bandpass_needed"]:
            bandpass = False
            skipped.append("bandpass (DC ratio {:.4f})".format(det["_dc_ratio"]))
        if not det["median_ref_needed"]:
            do_median_ref = False
            skipped.append("median-ref (ratio {:.4f})".format(det["_median_ratio"]))
        if skipped:
            print("  ℹ Auto-detect: skipping already-applied steps: " + ", ".join(skipped))

    # Step 1: Bandpass filter (must be done at original sfreq)
    if bandpass:
        data = bandpass_butterworth(
            data, sfreq,
            low=bandpass_low,
            high=bandpass_high,
            order=bandpass_order,
        )

    # Step 2: Notch filter for powerline (at original sfreq)
    if notch is not None:
        data = notch_filter(data, sfreq, freq=notch, quality=notch_quality)

    # Step 3: Median re-reference
    if do_median_ref:
        data = median_rereference(data)

    # Step 4: Resample to target
    if do_resample:
        data = resample(data, sfreq, target_sfreq)
        sfreq = target_sfreq

    return data, sfreq
