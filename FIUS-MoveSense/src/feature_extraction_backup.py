"""
FIUS-MoveSense: Feature Extraction Module V3 (Temporal)
========================================================
Key fix: Added TEMPORAL features comparing consecutive scans.
Movement causes scan-to-scan changes; standing still doesn't.
Keeps original zone + time + frequency features too.
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
    signal_abs = np.abs(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(signal_abs)
    return {
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


def extract_zone_features(signal):
    features = {}
    for zone_name, (start, end) in config.ECHO_ZONES.items():
        end = min(end, len(signal))
        chunk = signal[start:end]
        features[f"zone_{zone_name}_mean"] = np.mean(chunk)
        features[f"zone_{zone_name}_std"] = np.std(chunk)
    return features


def extract_frequency_features(signal):
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
    return {
        "dominant_freq": dominant_freq,
        "mean_freq": mean_freq,
        "band_energy_0_50Hz": band_energy,
        "spectral_entropy": spec_entropy,
        "frequency_centroid": centroid,
        "spectral_flatness": flatness,
    }


def extract_temporal_features(current_signal, prev_signal, next_signal):
    """Compare this scan to its neighbors.
    Movement = large differences. Standing still = nearly identical scans.
    """
    features = {}
    neighbors = []
    if prev_signal is not None:
        neighbors.append(prev_signal)
    if next_signal is not None:
        neighbors.append(next_signal)

    if len(neighbors) == 0:
        features["temporal_mean_abs_diff"] = 0.0
        features["temporal_max_abs_diff"] = 0.0
        features["temporal_correlation"] = 1.0
        features["temporal_energy_diff"] = 0.0
        features["temporal_std_diff"] = 0.0
        features["temporal_zone_close_diff"] = 0.0
        features["temporal_zone_mid_diff"] = 0.0
        features["temporal_zone_far_diff"] = 0.0
        features["temporal_zone_bg_diff"] = 0.0
        return features

    all_mean_diffs = []
    all_max_diffs = []
    all_corrs = []
    all_energy_diffs = []
    all_std_diffs = []
    all_zone_diffs = {z: [] for z in config.ECHO_ZONES}

    for neighbor in neighbors:
        diff = np.abs(current_signal - neighbor)
        all_mean_diffs.append(np.mean(diff))
        all_max_diffs.append(np.max(diff))
        corr = np.corrcoef(current_signal, neighbor)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        all_corrs.append(corr)
        e_curr = np.sum(current_signal ** 2)
        e_neigh = np.sum(neighbor ** 2)
        all_energy_diffs.append(abs(e_curr - e_neigh) / (e_curr + 1e-10))
        all_std_diffs.append(abs(np.std(current_signal) - np.std(neighbor)))
        for zone_name, (start, end) in config.ECHO_ZONES.items():
            end = min(end, len(current_signal))
            zone_diff = np.mean(np.abs(current_signal[start:end] - neighbor[start:end]))
            all_zone_diffs[zone_name].append(zone_diff)

    features["temporal_mean_abs_diff"] = np.mean(all_mean_diffs)
    features["temporal_max_abs_diff"] = np.mean(all_max_diffs)
    features["temporal_correlation"] = np.mean(all_corrs)
    features["temporal_energy_diff"] = np.mean(all_energy_diffs)
    features["temporal_std_diff"] = np.mean(all_std_diffs)
    features["temporal_zone_close_diff"] = np.mean(all_zone_diffs.get("close", [0]))
    features["temporal_zone_mid_diff"] = np.mean(all_zone_diffs.get("mid", [0]))
    features["temporal_zone_far_diff"] = np.mean(all_zone_diffs.get("far", [0]))
    features["temporal_zone_bg_diff"] = np.mean(all_zone_diffs.get("background", [0]))
    return features


def extract_all_features(signal, first_peak_idx, prev_signal=None, next_signal=None):
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_zone_features(signal))
    features.update(extract_frequency_features(signal))
    features.update(extract_temporal_features(signal, prev_signal, next_signal))
    features["first_peak_idx"] = first_peak_idx
    return features


def build_feature_matrix(filtered_signals, first_peaks, labels):
    """Build feature matrix with temporal context.
    Uses file_boundaries to avoid comparing scans across files.
    """
    print("Feature Extraction...")
    print(f"  Processing {filtered_signals.shape[0]} signals...")

    boundaries_path = os.path.join(config.PROCESSED_DIR, "file_boundaries.npy")
    if os.path.exists(boundaries_path):
        file_boundaries = np.load(boundaries_path)
        print(f"  Using file boundaries ({len(file_boundaries) - 1} files)")
    else:
        file_boundaries = np.array([0, filtered_signals.shape[0]])
        print("  No file boundaries found - treating as single sequence")

    all_features = []
    n = filtered_signals.shape[0]

    for i in range(n):
        file_idx = np.searchsorted(file_boundaries, i, side='right') - 1
        file_start = file_boundaries[file_idx]
        file_end = file_boundaries[file_idx + 1] if file_idx + 1 < len(file_boundaries) else n

        prev_sig = filtered_signals[i - 1] if i > file_start else None
        next_sig = filtered_signals[i + 1] if i < file_end - 1 else None

        feats = extract_all_features(
            filtered_signals[i], first_peaks[i],
            prev_signal=prev_sig, next_signal=next_sig
        )
        feats["label"] = labels[i]
        all_features.append(feats)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n} signals processed")

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
