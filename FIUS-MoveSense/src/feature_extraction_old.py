"""
FIUS-MoveSense: Feature Extraction Module V2
==============================================
Key fix: Replaced fixed echo zones with peak-adaptive zones.
Fixed zones made the model learn distance patterns instead of
movement patterns. Now zones are relative to where the echo is.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis, entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def find_echo_peak_region(signal, window=1000):
    """Find the center of the main echo in the signal."""
    abs_sig = np.abs(signal)
    energy = np.convolve(abs_sig ** 2, np.ones(window) / window, mode='valid')
    peak_center = np.argmax(energy) + window // 2
    peak_energy = energy[np.argmax(energy)]
    return int(peak_center), float(peak_energy)


def extract_time_features(signal):
    """Extract global time-domain statistical features."""
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


def extract_peak_adaptive_features(signal):
    """Extract features relative to the echo peak position.
    
    Zones are defined relative to where the echo actually is,
    making features distance-independent.
    """
    N = len(signal)
    peak_center, peak_energy = find_echo_peak_region(signal)
    half_echo = 2500

    echo_start = max(0, peak_center - half_echo)
    echo_end = min(N, peak_center + half_echo)
    pre_start = max(0, peak_center - 3 * half_echo)
    pre_end = echo_start
    post_start = echo_end
    post_end = min(N, peak_center + 3 * half_echo)

    echo_region = signal[echo_start:echo_end]
    pre_region = signal[pre_start:pre_end] if pre_end > pre_start else signal[0:100]
    post_region = signal[post_start:post_end] if post_end > post_start else signal[-100:]

    noise_before = signal[0:max(1, pre_start)]
    noise_after = signal[min(N - 1, post_end):]
    noise_region = np.concatenate([noise_before, noise_after]) if len(noise_before) + len(noise_after) > 0 else signal[0:100]

    features = {
        "echo_peak_position": peak_center / N,
        "echo_peak_energy": peak_energy,
        "echo_std": np.std(echo_region),
        "echo_energy": np.sum(echo_region ** 2) / len(echo_region),
        "echo_peak_to_peak": np.max(echo_region) - np.min(echo_region),
        "pre_echo_std": np.std(pre_region),
        "pre_echo_energy": np.sum(pre_region ** 2) / len(pre_region),
        "post_echo_std": np.std(post_region),
        "post_echo_energy": np.sum(post_region ** 2) / len(post_region),
        "noise_std": np.std(noise_region),
        "noise_energy": np.sum(noise_region ** 2) / len(noise_region),
        "echo_to_noise_ratio": np.std(echo_region) / (np.std(noise_region) + 1e-10),
        "echo_to_pre_ratio": np.std(echo_region) / (np.std(pre_region) + 1e-10),
        "echo_to_post_ratio": np.std(echo_region) / (np.std(post_region) + 1e-10),
        "echo_skewness": skew(echo_region) if len(echo_region) > 2 else 0,
        "echo_kurtosis": kurtosis(echo_region) if len(echo_region) > 2 else 0,
    }
    return features


def extract_frequency_features(signal):
    """Extract frequency-domain features from the FFT."""
    N = len(signal)
    spectrum = np.abs(fft(signal))[:N // 2]
    freqs = np.fft.fftfreq(N, d=1 / config.SAMPLING_RATE)[:N // 2]
    total_power = np.sum(spectrum)
    if total_power == 0:
        return {
            "dominant_freq": 0, "mean_freq": 0, "band_energy_0_50Hz": 0,
            "spectral_entropy": 0, "frequency_centroid": 0, "spectral_flatness": 0,
        }
    dominant_freq = freqs[np.argmax(spectrum)]
    norm_spectrum = spectrum / total_power
    mean_freq = np.sum(freqs * norm_spectrum)
    band_mask = freqs <= 50
    band_energy = np.sum(spectrum[band_mask])
    norm_safe = norm_spectrum[norm_spectrum > 0]
    spec_entropy = entropy(norm_safe) if len(norm_safe) > 0 else 0
    centroid = np.sum(freqs * spectrum) / total_power
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
    features.update(extract_peak_adaptive_features(signal))
    features.update(extract_frequency_features(signal))
    features["first_peak_idx"] = first_peak_idx
    return features


def build_feature_matrix(filtered_signals, first_peaks, labels):
    """Build complete feature matrix from all signals."""
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
    print(f"  Features: {df.shape[1] - 1}")

    feature_cols = [c for c in df.columns if c != "label"]
    desc_df = pd.DataFrame({"feature": feature_cols, "index": range(len(feature_cols))})
    desc_df.to_csv(os.path.join(config.FEATURES_DIR, "feature_descriptions.csv"), index=False)

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