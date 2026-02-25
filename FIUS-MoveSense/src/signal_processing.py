"""
FIUS-MoveSense: Signal Processing Module
==========================================
Applies Bessel lowpass filter, computes Hilbert envelope,
and detects echo peaks in each signal.
"""

import os
import sys
import numpy as np
from scipy.signal import bessel, filtfilt, hilbert, find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def apply_bessel_filter(signals):
    """Apply Bessel lowpass filter to all signals.
    
    Bessel filter preserves waveform shape (flat group delay).
    Same proven filter used in the old project.
    """
    b, a = bessel(N=config.BESSEL_ORDER, Wn=config.BESSEL_WN, btype='low', analog=False, norm='phase')
    filtered = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        filtered[i] = filtfilt(b, a, signals[i])
    return filtered


def compute_envelope(signals):
    """Compute signal envelope using Hilbert transform."""
    envelopes = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        analytic = hilbert(signals[i])
        envelopes[i] = np.abs(analytic)
    return envelopes


def detect_first_peak(signal, envelope=None):
    """Detect the index of the first significant echo peak.
    
    Uses the envelope if provided, otherwise uses the raw signal.
    Returns the sample index of the first peak, or -1 if none found.
    """
    if envelope is not None:
        data = envelope
    else:
        data = np.abs(signal)

    std = np.std(data)
    threshold = std * config.PEAK_THRESHOLD_FACTOR
    peaks, _ = find_peaks(data, height=threshold, distance=50)

    # Skip peaks in the very first samples (transmit ringing zone)
    valid_peaks = peaks[peaks > 500]

    if len(valid_peaks) > 0:
        return valid_peaks[0]
    elif len(peaks) > 0:
        return peaks[0]
    else:
        return -1


def process_signals(signals):
    """Full signal processing pipeline.
    
    Returns:
        filtered: Bessel-filtered signals
        envelopes: Hilbert envelope of filtered signals
        first_peaks: Index of first echo peak per signal
    """
    print("Signal Processing...")
    print(f"  Applying Bessel filter (order={config.BESSEL_ORDER}, Wn={config.BESSEL_WN})...")
    filtered = apply_bessel_filter(signals)

    print("  Computing Hilbert envelopes...")
    envelopes = compute_envelope(filtered)

    print("  Detecting first echo peaks...")
    first_peaks = np.array([
        detect_first_peak(filtered[i], envelopes[i])
        for i in range(filtered.shape[0])
    ])
    valid_count = np.sum(first_peaks >= 0)
    print(f"    -> {valid_count}/{len(first_peaks)} signals have valid peaks")

    # Save processed data
    np.save(os.path.join(config.PROCESSED_DIR, "filtered.npy"), filtered)
    np.save(os.path.join(config.PROCESSED_DIR, "envelopes.npy"), envelopes)
    np.save(os.path.join(config.PROCESSED_DIR, "first_peaks.npy"), first_peaks)
    print(f"  Saved to {config.PROCESSED_DIR}/")

    return filtered, envelopes, first_peaks


if __name__ == "__main__":
    print("Loading signals...")
    signals = np.load(os.path.join(config.PROCESSED_DIR, "signals.npy"))
    filtered, envelopes, first_peaks = process_signals(signals)
    print("Done!")
