"""
FIUS-MoveSense: Data Loading Module
====================================
Loads raw CSV sensor data, strips headers, handles split files,
and produces clean numpy arrays ready for processing.

Supports two modes:
  - "sample": loads the 3 original test files (line-wrapped format)
  - "final":  loads organized CSV files from data/raw/no_movement/ and data/raw/movement/
"""

import os
import sys
import numpy as np
# tqdm is optional - used on student's local machine
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _parse_clean_csv(filepath):
    """Parse a clean CSV where each row = one complete scan (50017 values)."""
    signals = []
    metadata = []

    # Try multiple encodings
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                f.readline()  # test read
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        encoding = 'latin-1'  # fallback that accepts any byte

    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(',')
            if len(values) < config.HEADER_FIELDS + 100:
                continue  # skip malformed rows

            try:
                meta = [float(v) for v in values[:config.HEADER_FIELDS]]
                sig = [float(v) for v in values[config.HEADER_FIELDS:config.HEADER_FIELDS + config.EXPECTED_SIGNAL_LENGTH]]
            except ValueError:
                continue

            # Pad or truncate signal to expected length
            if len(sig) < config.EXPECTED_SIGNAL_LENGTH:
                sig.extend([0.0] * (config.EXPECTED_SIGNAL_LENGTH - len(sig)))
            sig = sig[:config.EXPECTED_SIGNAL_LENGTH]

            metadata.append(meta)
            signals.append(sig)

    return np.array(signals, dtype=np.float64), metadata


def _parse_split_csv(filepaths):
    """Parse CSV files where scans may be split across lines or files.
    
    The movement sample files have lines split at exactly 32,759 byte boundaries,
    which can split in the middle of numbers. Strategy: join all text, then use
    regex to restore missing commas at split points.
    """
    import re
    
    # Read all lines from all files
    full_text = ""
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8-sig') as f:
            for line in f:
                full_text += line.rstrip('\n').rstrip('\r')
    
    # Fix fused values caused by line-break splits:
    # Case 1: "-253.0-184.0" -> "-253.0,-184.0" (digit before minus sign)
    full_text = re.sub(r'(\d)-', r'\1,-', full_text)
    # Case 2: "84.0114.0" -> "84.0,114.0" (ends .0 then starts new positive number)
    full_text = re.sub(r'\.0(\d)', r'.0,\1', full_text)

    # Split by header pattern to identify individual scans
    # The header pattern "68.0,50000.0,0.0,1.0,512.0," consumes the first 5 header fields
    # Each part then starts with the remaining 12 header fields + 50000 signal samples
    HEADER_CONSUMED = 5  # fields consumed by the split pattern
    HEADER_REMAINING = config.HEADER_FIELDS - HEADER_CONSUMED  # 12 fields left

    parts = full_text.split(config.HEADER_PATTERN)
    parts = [p for p in parts if p.strip()]

    signals = []
    metadata = []
    min_signal_samples = config.EXPECTED_SIGNAL_LENGTH - 1000

    for part in parts:
        values = part.split(',')
        values = [v.strip() for v in values if v.strip()]

        total_signal = len(values) - HEADER_REMAINING
        if total_signal < min_signal_samples:
            continue  # skip incomplete scans (boundary fragments)

        try:
            # Reconstruct full header
            consumed_header = [68.0, 50000.0, 0.0, 1.0, 512.0]
            remaining_header = [float(v) for v in values[:HEADER_REMAINING]]
            meta = consumed_header + remaining_header

            # Extract signal data
            sig = [float(v) for v in values[HEADER_REMAINING:HEADER_REMAINING + config.EXPECTED_SIGNAL_LENGTH]]
        except (ValueError, IndexError):
            continue

        if len(sig) < config.EXPECTED_SIGNAL_LENGTH:
            sig.extend([0.0] * (config.EXPECTED_SIGNAL_LENGTH - len(sig)))
        sig = sig[:config.EXPECTED_SIGNAL_LENGTH]

        metadata.append(meta)
        signals.append(sig)

    return np.array(signals, dtype=np.float64), metadata


def load_sample_data():
    """Load the sample/test data files."""
    print("Loading sample data...")

    # No movement - clean format
    nm_file = config.SAMPLE_DATA["no_movement"][0]
    print(f"  Loading no_movement: {os.path.basename(nm_file)}")
    nm_signals, nm_meta = _parse_clean_csv(nm_file)
    print(f"    -> {nm_signals.shape[0]} scans loaded")

    # Movement - split format across 2 files
    mv_files = config.SAMPLE_DATA["movement"]
    print(f"  Loading movement: {[os.path.basename(f) for f in mv_files]}")
    mv_signals, mv_meta = _parse_split_csv(mv_files)
    print(f"    -> {mv_signals.shape[0]} scans loaded")

    # Create labels
    nm_labels = np.zeros(nm_signals.shape[0], dtype=np.int32)
    mv_labels = np.ones(mv_signals.shape[0], dtype=np.int32)

    # Combine
    all_signals = np.vstack([nm_signals, mv_signals])
    all_labels = np.concatenate([nm_labels, mv_labels])

    return all_signals, all_labels


def load_final_data():
    """Load the final organized data from folders."""
    print("Loading final data...")
    all_signals = []
    all_labels = []

    for class_name, folder in config.FINAL_DATA.items():
        label = config.LABEL_MAP[class_name]
        if not os.path.exists(folder):
            print(f"  WARNING: Folder not found: {folder}")
            continue

        csv_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.endswith('.csv')
        ])
        print(f"  Loading {class_name}: {len(csv_files)} files")

        for fpath in tqdm(csv_files, desc=f"    {class_name}", leave=False):
            signals, meta = _parse_clean_csv(fpath)
            labels = np.full(signals.shape[0], label, dtype=np.int32)
            all_signals.append(signals)
            all_labels.append(labels)
            print(f"    {os.path.basename(fpath)}: {signals.shape[0]} scans")

    all_signals = np.vstack(all_signals)
    all_labels = np.concatenate(all_labels)

    return all_signals, all_labels


def load_data():
    """Load data based on current DATA_MODE in config."""
    if config.DATA_MODE == "sample":
        signals, labels = load_sample_data()
    elif config.DATA_MODE == "final":
        signals, labels = load_final_data()
    else:
        raise ValueError(f"Unknown DATA_MODE: {config.DATA_MODE}")

    print(f"\nDataset Summary:")
    print(f"  Total scans: {signals.shape[0]}")
    print(f"  Signal length: {signals.shape[1]} samples")
    print(f"  Not Moving (0): {np.sum(labels == 0)}")
    print(f"  Moving (1):     {np.sum(labels == 1)}")

    # Save processed arrays for fast reloading
    np.save(os.path.join(config.PROCESSED_DIR, "signals.npy"), signals)
    np.save(os.path.join(config.PROCESSED_DIR, "labels.npy"), labels)
    print(f"\n  Saved to {config.PROCESSED_DIR}/")

    return signals, labels


if __name__ == "__main__":
    signals, labels = load_data()