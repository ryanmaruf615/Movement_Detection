"""
FIUS-MoveSense: Feature Extraction Module
===========================================
Extracts time-domain, windowed, and frequency-domain features
from processed signals. Produces a feature matrix for ML training.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis, entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_time_features(signal):
    """Extract global time-domain statistical features from one signal."""
    signal_abs = np.abs(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(signal_abs)

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": rms,
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "energy": np.sum(signal ** 2),
        "peak_to_peak": np.max(signal) - np.min(signal),
        "crest_factor": peak / rms if rms > 0 else 0,
        "zero_crossing_rate": np.sum(np.diff(np.sign(signal)) != 0) / len(signal),
        "percentile_25": np.percentile(signal, 25),
        "percentile_50": np.percentile(signal, 50),
        "percentile_75": np.percentile(signal, 75),
    }
    return features


def extract_zone_features(signal):
    """Extract mean and std for each physically meaningful echo zone."""
    features = {}
    for zone_name, (start, end) in config.ECHO_ZONES.items():
        end = min(end, len(signal))
        chunk = signal[start:end]
        features[f"zone_{zone_name}_mean"] = np.mean(chunk)
        features[f"zone_{zone_name}_std"] = np.std(chunk)
    return features


def extract_frequency_features(signal):
    """Extract frequency-domain features from the FFT of the signal."""
    N = len(signal)
    spectrum = np.abs(fft(signal))[:N // 2]
    freqs = np.fft.fftfreq(N, d=1 / config.SAMPLING_RATE)[:N // 2]

    total_power = np.sum(spectrum)
    if total_power == 0:
        total_power = 1e-10

    norm_spectrum = spectrum / total_power

    # Dominant frequency (excluding DC component)
    dominant_freq = freqs[1:][np.argmax(spectrum[1:])] if len(spectrum) > 1 else 0

    # Mean frequency (weighted)
    mean_freq = np.sum(freqs * norm_spectrum)

    # Band energy 0-50 Hz
    band_mask = (freqs >= 0) & (freqs <= 50)
    band_energy = np.sum(spectrum[band_mask])

    # Spectral entropy
    norm_safe = norm_spectrum[norm_spectrum > 0]
    spec_entropy = entropy(norm_safe) if len(norm_safe) > 0 else 0

    # Frequency centroid
    centroid = np.sum(freqs * spectrum) / total_power

    # Spectral flatness
    log_mean = np.mean(np.log(spectrum + 1e-10))
    arith_mean = np.mean(spectrum) + 1e-10
    flatness = np.exp(log_mean) / arith_mean

    features = {
        "dominant_freq": dominant_freq,
        "mean_freq": mean_freq,
        "band_energy_0_50Hz": band_energy,
        "spectral_entropy": spec_entropy,
        "frequency_centroid": centroid,
        "spectral_flatness": flatness,
    }
    return features


def extract_all_features(signal, first_peak_idx):
    """Extract all features from one signal."""
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_zone_features(signal))
    features.update(extract_frequency_features(signal))
    features["first_peak_idx"] = first_peak_idx
    return features


def build_feature_matrix(filtered_signals, first_peaks, labels):
    """Build complete feature matrix from all signals.
    
    Returns:
        DataFrame with features and labels
    """
    print("Feature Extraction...")
    print(f"  Processing {filtered_signals.shape[0]} signals...")

    all_features = []
    for i in range(filtered_signals.shape[0]):
        feats = extract_all_features(filtered_signals[i], first_peaks[i])
        feats["label"] = labels[i]
        all_features.append(feats)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{filtered_signals.shape[0]} signals processed")

    df = pd.DataFrame(all_features)
    print(f"  Feature matrix shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 1}")  # minus label column

    # Save feature descriptions
    feature_cols = [c for c in df.columns if c != "label"]
    desc_df = pd.DataFrame({
        "feature": feature_cols,
        "index": range(len(feature_cols))
    })
    desc_df.to_csv(os.path.join(config.FEATURES_DIR, "feature_descriptions.csv"), index=False)

    # Save feature matrix
    output_path = os.path.join(config.FEATURES_DIR, "features.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    return df


if __name__ == "__main__":
    print("Loading processed data...")
    filtered = np.load(os.path.join(config.PROCESSED_DIR, "filtered.npy"))
    first_peaks = np.load(os.path.join(config.PROCESSED_DIR, "first_peaks.npy"))
    labels = np.load(os.path.join(config.PROCESSED_DIR, "labels.npy"))
    df = build_feature_matrix(filtered, first_peaks, labels)
    print("Done!")
