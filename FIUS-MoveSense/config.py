"""
FIUS-MoveSense: Central Configuration
=====================================
All paths, parameters, and settings in one place.
Change DATA_MODE to switch between sample and final data.
"""

import os

# ============================================================
# DATA MODE: Change this when switching to final data
# ============================================================
# "sample"  -> uses the 3 sample files (signal_500/600)
# "final"   -> uses files in data/raw/no_movement/ and data/raw/movement/
DATA_MODE = "final"

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sample data paths (the 3 files uploaded earlier)
SAMPLE_DATA = {
    "no_movement": [
        os.path.join(BASE_DIR, "data", "raw", "signal_500_distance_1_5_meter_no_movement.csv"),
    ],
    "movement": [
        os.path.join(BASE_DIR, "data", "raw", "signal_600_distance_1_5_meter_with_movement_part1.csv"),
        os.path.join(BASE_DIR, "data", "raw", "signal_600_distance_1_5_meter_with_movement_part2.csv"),
    ],
}

# Final data paths (organized folders)
FINAL_DATA = {
    "no_movement": os.path.join(BASE_DIR, "data", "raw", "no_movement"),
    "movement": os.path.join(BASE_DIR, "data", "raw", "movement"),
}

# Output paths
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
SIGNAL_PLOTS_DIR = os.path.join(RESULTS_DIR, "signal_examples")

# Create directories if they don't exist
for d in [PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, CONFUSION_DIR, SIGNAL_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# SENSOR PARAMETERS
# ============================================================
SAMPLING_RATE = 1_953_125          # Hz
SPEED_OF_SOUND = 343.2             # m/s at ~20°C
HEADER_FIELDS = 17                 # metadata columns per scan
EXPECTED_SIGNAL_LENGTH = 50_000    # ADC samples per scan
HEADER_PATTERN = "68.0,50000.0,0.0,1.0,512.0,"  # for reassembling split files

# ============================================================
# SIGNAL PROCESSING PARAMETERS
# ============================================================
BESSEL_ORDER = 2
BESSEL_WN = 0.6                    # normalized cutoff frequency
PEAK_THRESHOLD_FACTOR = 0.2        # peak height = factor × std

# Echo zones (physically meaningful distance regions)
# Zone boundaries in sample indices
ECHO_ZONES = {
    "near_field":  (0, 5000),       # 0 - 0.44 m (TX ringing)
    "close":       (5000, 11000),    # 0.44 - 0.97 m
    "mid":         (11000, 17000),   # 0.97 - 1.50 m
    "far":         (17000, 25000),   # 1.50 - 2.20 m
    "background":  (25000, 50000),   # 2.20 - 4.39 m
}

# ============================================================
# ML PARAMETERS
# ============================================================
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
CV_FOLDS = 5

# Labels
LABEL_MAP = {"no_movement": 0, "movement": 1}
LABEL_NAMES = {0: "Not Moving", 1: "Moving"}