"""
FIUS-MoveSense: Complete Fix Script
====================================
Run this ONCE to:
1. Update feature_extraction.py with temporal features
2. Update data_loading.py with file boundaries
3. Update predict.py with temporal context
4. Clean up old models

Usage:
    python fix_all.py

After running, do:
    python main.py
    python -c "import pandas as pd; df=pd.read_csv('data/features/features.csv'); cols=[c for c in df.columns if c!='label']; f=open('models/feature_order.txt','w'); [f.write(c+'\n') for c in cols]; f.close(); print('Done:', len(cols), 'features')"
    python predict.py data\raw\test\T1.csv
"""

import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")
MODELS = os.path.join(BASE, "models")

print("=" * 60)
print("FIUS-MoveSense: Applying Temporal Feature Fix")
print("=" * 60)

# ============================================================
# 1. BACKUP OLD FILES
# ============================================================
for f in ["feature_extraction.py", "data_loading.py"]:
    src = os.path.join(SRC, f)
    bak = os.path.join(SRC, f.replace(".py", "_backup.py"))
    if os.path.exists(src) and not os.path.exists(bak):
        shutil.copy2(src, bak)
        print(f"  Backed up {f}")

pred_src = os.path.join(BASE, "predict.py")
pred_bak = os.path.join(BASE, "predict_backup.py")
if os.path.exists(pred_src) and not os.path.exists(pred_bak):
    shutil.copy2(pred_src, pred_bak)
    print("  Backed up predict.py")

# ============================================================
# 2. WRITE NEW feature_extraction.py
# ============================================================
FE_CODE = '''"""
FIUS-MoveSense: Feature Extraction Module V3 (Temporal)
========================================================
Adds TEMPORAL features comparing consecutive scans.
Movement causes scan-to-scan changes; standing still doesn't.
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
        "dominant_freq": dominant_freq, "mean_freq": mean_freq,
        "band_energy_0_50Hz": band_energy, "spectral_entropy": spec_entropy,
        "frequency_centroid": centroid, "spectral_flatness": flatness,
    }


def extract_temporal_features(current_signal, prev_signal, next_signal):
    """Compare this scan to neighbors. Movement = large diffs."""
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
'''

fe_path = os.path.join(SRC, "feature_extraction.py")
with open(fe_path, 'w') as f:
    f.write(FE_CODE)
print(f"\n  Written: {fe_path}")

# Verify
with open(fe_path) as f:
    content = f.read()
assert "temporal" in content, "ERROR: temporal not found in feature_extraction.py!"
print("  Verified: temporal features present")


# ============================================================
# 3. UPDATE data_loading.py - add file_boundaries saving
# ============================================================
dl_path = os.path.join(SRC, "data_loading.py")
with open(dl_path, 'r') as f:
    dl_code = f.read()

if "file_boundaries" not in dl_code:
    # Replace load_final_data to add boundary tracking
    old_final = '''    all_signals = np.vstack(all_signals)
    all_labels = np.concatenate(all_labels)

    return all_signals, all_labels'''
    
    new_final = '''    all_signals = np.vstack(all_signals)
    all_labels = np.concatenate(all_labels)

    # Save file boundaries for temporal feature extraction
    np.save(os.path.join(config.PROCESSED_DIR, "file_boundaries.npy"),
            np.array(file_boundaries, dtype=np.int64))
    print(f"  File boundaries saved ({len(file_boundaries) - 1} files)")

    return all_signals, all_labels'''
    
    old_init = '''    all_signals = []
    all_labels = []

    for class_name'''
    
    new_init = '''    all_signals = []
    all_labels = []
    file_boundaries = [0]  # track where each file starts/ends

    for class_name'''
    
    old_append = '''            all_signals.append(signals)
            all_labels.append(labels)
            print'''
    
    new_append = '''            all_signals.append(signals)
            all_labels.append(labels)
            file_boundaries.append(file_boundaries[-1] + signals.shape[0])
            print'''
    
    if old_final in dl_code:
        dl_code = dl_code.replace(old_final, new_final)
        dl_code = dl_code.replace(old_init, new_init, 1)
        dl_code = dl_code.replace(old_append, new_append, 1)
        
        with open(dl_path, 'w') as f:
            f.write(dl_code)
        print(f"\n  Updated: {dl_path}")
        print("  Added: file_boundaries tracking")
    else:
        print(f"\n  WARNING: Could not patch {dl_path}")
        print("  The file format may have changed. Manual update needed.")
else:
    print(f"\n  {dl_path} already has file_boundaries - skipping")


# ============================================================
# 4. UPDATE predict.py - add temporal context
# ============================================================
pred_path = os.path.join(BASE, "predict.py")
with open(pred_path, 'r') as f:
    pred_code = f.read()

if "prev_sig" not in pred_code:
    old_extract = '''    print("Extracting features...")
    all_features = []
    for i in range(filtered.shape[0]):
        first_peak = detect_first_peak(filtered[i], envelopes[i])
        feats = extract_all_features(filtered[i], first_peak)
        # Use training feature order, NOT sorted
        all_features.append([feats[k] for k in feature_order])

        if (i + 1) % 200 == 0:
            print("  {}/{} scans processed".format(i + 1, signals.shape[0]))'''
    
    new_extract = '''    print("Extracting features...")
    all_features = []
    n = filtered.shape[0]
    for i in range(n):
        first_peak = detect_first_peak(filtered[i], envelopes[i])
        prev_sig = filtered[i - 1] if i > 0 else None
        next_sig = filtered[i + 1] if i < n - 1 else None
        feats = extract_all_features(filtered[i], first_peak,
                                     prev_signal=prev_sig, next_signal=next_sig)
        # Use training feature order, NOT sorted
        all_features.append([feats[k] for k in feature_order])

        if (i + 1) % 200 == 0:
            print("  {}/{} scans processed".format(i + 1, signals.shape[0]))'''
    
    if old_extract in pred_code:
        pred_code = pred_code.replace(old_extract, new_extract)
        with open(pred_path, 'w') as f:
            f.write(pred_code)
        print(f"\n  Updated: {pred_path}")
        print("  Added: temporal context (prev_sig, next_sig)")
    else:
        print(f"\n  WARNING: Could not patch {pred_path}")
        print("  The extract section format may have changed.")
else:
    print(f"\n  {pred_path} already has temporal context - skipping")


# ============================================================
# 5. CLEAN OLD MODELS
# ============================================================
print("\n  Cleaning old models...")
for f in os.listdir(MODELS):
    fpath = os.path.join(MODELS, f)
    if f.endswith('.joblib') or f == 'feature_order.txt':
        os.remove(fpath)
        print(f"    Deleted: {f}")

# Remove old file_boundaries if exists
fb_path = os.path.join(BASE, "data", "processed", "file_boundaries.npy")
if os.path.exists(fb_path):
    os.remove(fb_path)
    print("    Deleted: file_boundaries.npy")


# ============================================================
# 6. VERIFY
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

with open(os.path.join(SRC, "feature_extraction.py")) as f:
    c = f.read()
print(f"  feature_extraction.py has 'temporal': {'temporal' in c}")

with open(os.path.join(SRC, "data_loading.py")) as f:
    c = f.read()
print(f"  data_loading.py has 'file_boundaries': {'file_boundaries' in c}")

with open(os.path.join(BASE, "predict.py")) as f:
    c = f.read()
print(f"  predict.py has 'prev_sig': {'prev_sig' in c}")

print("\n" + "=" * 60)
print("FIX APPLIED SUCCESSFULLY!")
print("=" * 60)
print("\nNow run:")
print("  python main.py")
print('  python -c "import pandas as pd; df=pd.read_csv(\'data/features/features.csv\'); cols=[c for c in df.columns if c!=\'label\']; f=open(\'models/feature_order.txt\',\'w\'); [f.write(c+\'\\n\') for c in cols]; f.close(); print(\'Done:\', len(cols), \'features\')"')
print("  python predict.py data\\raw\\test\\T1.csv")
