
# # Fall Detection — Time Series Classification Pipeline
# 
# **Dataset:** IMU sensor data (AccX/Y/Z, GyrX/Y/Z, EulerX/Y/Z) @ 100 Hz
# **Task:** Binary classification — Fall (1) vs No-Fall (0)
# **Balancing methods:**
# - Method 1: Random Undersampling of majority class
# - Method 2: NearMiss-1 Undersampling (boundary-aware)
# 
# **Pipeline:** Sliding Window → Feature Extraction → 4 Classifiers × 3 dataset variants

# In[1]:
# =============================================================
# CELL 1 — Imports & Global Configuration
# =============================================================
import os, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — GradientBoosting used for classifier 4")

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Colab/session-storage aware project root resolution
IS_COLAB = ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_GPU" in os.environ)
if IS_COLAB:
    # User requested Colab session storage layout: /content/DataMiners/data
    PROJECT_ROOT = Path("/content/DataMiners")
else:
    # main.py lives at the project root — resolve relative to this file
    # so the script works regardless of which directory you run it from.
    PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}. "
        "In Colab, make sure the repo is cloned to /content/DataMiners."
    )


CONFIG = {
    "PROJECT_ROOT"   : PROJECT_ROOT,
    "DATA_DIR"       : DATA_DIR,
    "TRAIN_DIR"      : DATA_DIR / "Sample_Training",
    "TEST_DIR"       : DATA_DIR / "Sample_Test",
    "FEATURE_COLS"   : ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ","EulerX","EulerY","EulerZ"],
    "LABEL_COL"      : "FallCheck",
    "WINDOW_SIZE"    : 50,
    "STEP_SIZE"      : 25,        # 50% overlap
    "FALL_THRESHOLD" : 0.40,      # >40% ones => window label = 1
    "N_FOLDS"        : 3,         # Plan A: 5→3-fold outer CV (-40% CV time)
    "RANDOM_STATE"   : 42,
    "N_JOBS"         : -1,
    "OUTPUT_DIR"     : DATA_DIR / "Balanced",
}
CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

print("Configuration ready.")
print(f"  Runtime: {'Colab' if IS_COLAB else 'Local/Jupyter'}")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# In[2]:
# =============================================================
# CELL 2 — Data Loading
# =============================================================

def load_subject_csvs(subject_dir: Path, feature_cols: list, label_col: str) -> pd.DataFrame:
    """Load and concatenate all CSVs for one subject folder."""
    required = feature_cols + [label_col]
    frames = []
    for csv_path in sorted(subject_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in required)
            if all(c in df.columns for c in required) and len(df) > 0:
                frames.append(df[required].reset_index(drop=True))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required)


def load_dataset(base_dir: Path, feature_cols: list, label_col: str) -> dict:
    """Load all subject sub-folders from a dataset directory.
    Returns dict mapping subject_id -> DataFrame.
    """
    base_dir = Path(base_dir)
    subject_data, total_rows, total_files = {}, 0, 0
    for subj_dir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        total_files += len(list(subj_dir.glob("*.csv")))
        df = load_subject_csvs(subj_dir, feature_cols, label_col)
        if len(df) > 0:
            subject_data[subj_dir.name] = df
            total_rows += len(df)
    print(f"  {len(subject_data)} subjects | {total_files} files | {total_rows:,} rows")
    return subject_data


print("Loading training data ...")
train_data = load_dataset(CONFIG["TRAIN_DIR"], CONFIG["FEATURE_COLS"], CONFIG["LABEL_COL"])
print("Loading test data ...")
test_data  = load_dataset(CONFIG["TEST_DIR"],  CONFIG["FEATURE_COLS"], CONFIG["LABEL_COL"])

# In[3]:
# =============================================================
# CELL 3 — Sliding Window
# =============================================================

def apply_sliding_window(df, window_size, step_size, feature_cols, label_col, fall_threshold):
    """Slide a window of length window_size with step_size over a DataFrame.
    Label = 1 if fraction of 1s in window > fall_threshold, else 0.
    Returns: windows (N, window_size, n_features), labels (N,)
    """
    n_rows = len(df)
    if n_rows < window_size:
        return (np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=int))
    feat_arr  = df[feature_cols].values.astype(np.float32)
    label_arr = df[label_col].values.astype(int)
    windows, labels = [], []
    for s in range(0, n_rows - window_size + 1, step_size):
        windows.append(feat_arr[s : s + window_size])
        win_lbl = 1 if label_arr[s : s + window_size].sum() / window_size > fall_threshold else 0
        labels.append(win_lbl)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=int)


def window_dataset(subject_data, config):
    """Apply sliding window across all subjects. Returns (windows, labels)."""
    win_list, lbl_list = [], []
    for subj_id, df in subject_data.items():
        w, l = apply_sliding_window(
            df, config["WINDOW_SIZE"], config["STEP_SIZE"],
            config["FEATURE_COLS"], config["LABEL_COL"], config["FALL_THRESHOLD"])
        if len(w) > 0:
            win_list.append(w); lbl_list.append(l)
    all_w = np.concatenate(win_list)
    all_l = np.concatenate(lbl_list)
    cnt = Counter(all_l.tolist())
    print(f"  Windows: {len(all_l):,}  Class 0: {cnt[0]:,}  "
          f"Class 1: {cnt[1]:,}  Ratio: {cnt[0]/max(cnt[1],1):.1f}:1")
    return all_w, all_l


print("Applying sliding window to training data ...")
train_windows, train_labels = window_dataset(train_data, CONFIG)
print("Applying sliding window to test data ...")
test_windows,  test_labels  = window_dataset(test_data,  CONFIG)
print(f"\nWindow shapes — train: {train_windows.shape}  test: {test_windows.shape}")
print(f"Expected windows from a 729-row file: {(729-50)//25+1}")

# In[4]:
# =============================================================
# CELL 4 — Feature Extraction (written from scratch)
# =============================================================

STAT_NAMES = ["mean","std","min","max","range","rms","energy",
              "variance","skewness","kurtosis","zcr","mad","p25","p75","iqr"]


def _skewness(x):
    """Pearson moment skewness (biased)."""
    mu, sig = x.mean(), x.std()
    return float(np.mean(((x - mu) / sig) ** 3)) if sig > 1e-10 else 0.0


def _kurtosis(x):
    """Excess kurtosis / Fisher's definition (biased)."""
    mu, sig = x.mean(), x.std()
    return float(np.mean(((x - mu) / sig) ** 4) - 3.0) if sig > 1e-10 else 0.0


def _zcr(x):
    """Zero-crossing rate: fraction of sign-change pairs."""
    return float(np.sum(np.abs(np.diff(np.sign(x)))) / 2 / max(len(x) - 1, 1))


def compute_window_features(window: np.ndarray) -> np.ndarray:
    """Compute 15 hand-crafted statistics for each of 9 sensor channels.
    All formulas implemented manually — no library calls for statistics.
    Args:   window  shape (window_size, 9)
    Returns: np.ndarray shape (135,)  [15 stats x 9 channels]
    """
    feats = []
    for ch in range(window.shape[1]):
        x   = window[:, ch].astype(np.float64)
        mu  = x.mean(); sig = x.std()
        p25 = float(np.percentile(x, 25))
        p75 = float(np.percentile(x, 75))
        feats += [
            mu,                              # mean
            sig,                             # std
            float(x.min()),                  # min
            float(x.max()),                  # max
            float(x.max() - x.min()),        # range
            float(np.sqrt(np.mean(x ** 2))), # RMS = sqrt(mean(x^2))
            float(np.sum(x ** 2)),           # energy = sum(x^2)
            float(sig ** 2),                 # variance
            _skewness(x),                    # skewness
            _kurtosis(x),                    # excess kurtosis
            _zcr(x),                         # zero-crossing rate
            float(np.mean(np.abs(x - mu))), # MAD
            p25, p75, float(p75 - p25),      # p25, p75, IQR
        ]
    return np.array(feats, dtype=np.float32)


def get_feature_names(feature_cols: list, stat_names: list) -> list:
    """Build names like 'AccX_mean', 'GyrY_kurtosis', etc."""
    return [f"{col}_{stat}" for col in feature_cols for stat in stat_names]


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Vectorised feature extraction — NO Python per-window loop.
    All 15 statistics computed with axis-wise NumPy reductions over the
    full (N, window_size, 9) array, giving a ~40-80x speed-up vs a
    Python list comprehension on 49 k windows.

    windows: (N, window_size, 9)  →  X: (N, 135)
    """
    W = windows.astype(np.float64)          # (N, T, C)
    N, T, C = W.shape

    # ── pre-compute building blocks ─────────────────────────────────
    mu      = W.mean(axis=1)                # (N, C)  mean
    sig     = W.std(axis=1)                 # (N, C)  std  (biased)
    var     = sig ** 2                      # (N, C)  variance
    mn      = W.min(axis=1)                 # (N, C)  min
    mx      = W.max(axis=1)                 # (N, C)  max
    rng     = mx - mn                       # (N, C)  range
    rms     = np.sqrt((W ** 2).mean(axis=1))# (N, C)  RMS
    energy  = (W ** 2).sum(axis=1)          # (N, C)  energy
    p25     = np.percentile(W, 25, axis=1)  # (N, C)  25th percentile
    p75     = np.percentile(W, 75, axis=1)  # (N, C)  75th percentile
    iqr     = p75 - p25                     # (N, C)  IQR

    # MAD  = mean(|x - mean(x)|) per channel
    mad     = np.abs(W - mu[:, None, :]).mean(axis=1)  # (N, C)

    # Skewness  = mean(((x-mu)/sig)^3)  —  safe divide avoids /0
    sig_safe = np.where(sig > 1e-10, sig, 1.0)
    z        = (W - mu[:, None, :]) / sig_safe[:, None, :]  # (N, T, C)
    skew     = (z ** 3).mean(axis=1)                         # (N, C)
    skew     = np.where(sig > 1e-10, skew, 0.0)

    # Excess kurtosis  = mean(z^4) - 3
    kurt     = (z ** 4).mean(axis=1) - 3.0                   # (N, C)
    kurt     = np.where(sig > 1e-10, kurt, 0.0)

    # ZCR  = (number of sign changes) / 2 / (T-1)
    signs    = np.sign(W)                                     # (N, T, C)
    zcr      = (np.abs(np.diff(signs, axis=1)).sum(axis=1)
                / 2 / max(T - 1, 1))                         # (N, C)

    # ── stack all 15 stats along the last axis, then flatten ────────
    # Order must match STAT_NAMES: mean std min max range rms energy
    #   variance skewness kurtosis zcr mad p25 p75 iqr
    X = np.stack([
        mu, sig, mn, mx, rng, rms, energy,
        var, skew, kurt, zcr, mad, p25, p75, iqr,
    ], axis=2)                                                # (N, C, 15)

    return X.reshape(N, -1).astype(np.float32)               # (N, 135)


FEATURE_NAMES = get_feature_names(CONFIG["FEATURE_COLS"], STAT_NAMES)
assert len(FEATURE_NAMES) == 135

print("Extracting features from training windows ...")
X_train = extract_features(train_windows)
y_train = train_labels.copy()
print(f"  X_train: {X_train.shape}")

print("Extracting features from test windows ...")
X_test = extract_features(test_windows)
y_test  = test_labels.copy()
print(f"  X_test : {X_test.shape}")

del train_windows, test_windows
print(f"\n135 features = {len(CONFIG['FEATURE_COLS'])} channels x {len(STAT_NAMES)} statistics")

# In[5]:
# =============================================================
# CELL 5 — Custom Evaluation Functions (written from scratch)
# =============================================================

EPS = 1e-9


def compute_confusion_values(y_true, y_pred) -> dict:
    """Manually compute TP, TN, FP, FN — no sklearn dependency."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "TP": int(np.sum((y_true == 1) & (y_pred == 1))),
        "TN": int(np.sum((y_true == 0) & (y_pred == 0))),
        "FP": int(np.sum((y_true == 0) & (y_pred == 1))),
        "FN": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Compute TP, TN, FP, FN, F1 per class, F1 macro, balanced accuracy, accuracy."""
    cm  = compute_confusion_values(y_true, y_pred)
    TP, TN, FP, FN = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    prec1 = TP / (TP + FP + EPS);  rec1 = TP / (TP + FN + EPS)  # class-1 metrics
    prec0 = TN / (TN + FN + EPS);  rec0 = TN / (TN + FP + EPS)  # class-0 metrics
    f1_1  = 2 * prec1 * rec1 / (prec1 + rec1 + EPS)
    f1_0  = 2 * prec0 * rec0 / (prec0 + rec0 + EPS)
    return {
        **cm,
        "precision_1"      : round(prec1, 4),
        "recall_1"         : round(rec1,  4),   # sensitivity / TPR
        "precision_0"      : round(prec0, 4),
        "recall_0"         : round(rec0,  4),   # specificity / TNR
        "f1_class_0"       : round(f1_0,  4),
        "f1_class_1"       : round(f1_1,  4),
        "f1_macro"         : round((f1_0 + f1_1) / 2, 4),
        "balanced_accuracy": round((rec1 + rec0) / 2, 4),
        "accuracy"         : round((TP + TN) / max(TP + TN + FP + FN, 1), 4),
    }


def format_confusion_table(cm_dict: dict, model_name: str = "") -> pd.DataFrame:
    """Return a labelled 2×2 confusion matrix DataFrame."""
    df = pd.DataFrame(
        [[cm_dict["TN"], cm_dict["FP"]], [cm_dict["FN"], cm_dict["TP"]]],
        index   = ["Actual 0 (No-Fall)", "Actual 1 (Fall)"],
        columns = ["Predicted 0 (No-Fall)", "Predicted 1 (Fall)"],
    )
    if model_name:
        df.index.name = model_name
    return df


def format_metrics_table(metrics: dict, model_name: str = "") -> pd.DataFrame:
    """Return a single-row DataFrame of key metrics."""
    keys = ["TP","TN","FP","FN","f1_class_0","f1_class_1",
            "f1_macro","balanced_accuracy","accuracy"]
    df = pd.DataFrame([{k: metrics[k] for k in keys}])
    if model_name:
        df.insert(0, "Model", model_name)
    return df


# Unit test
_t, _p = np.array([0,0,1,1]), np.array([0,1,1,0])
_cm = compute_confusion_values(_t, _p)
assert _cm == {"TP":1,"TN":1,"FP":1,"FN":1}, f"Unit test failed: {_cm}"
print("Evaluation unit test passed: TP=1, TN=1, FP=1, FN=1")
print(format_metrics_table(compute_metrics(_t, _p)).to_string(index=False))

# In[6]:
# =============================================================
# CELL 6 — Imbalance Handling & Save Balanced Datasets
#
# Constraints strictly followed:
#   [x] Minority class ('1') samples are NEVER removed
#   [x] Minority class is NEVER upsampled (no SMOTE)
#
# Method 1: Random Undersampling (RUS) — randomly discard majority samples
# Method 2: NearMiss-1 Undersampling  — keep majority samples closest to minority
# =============================================================

def check_imbalance_ratio(y, split_name="") -> dict:
    """Print class distribution and imbalance ratio."""
    cnt = Counter(y.tolist()); total = len(y)
    tag = f"[{split_name}] " if split_name else ""
    print(f"{tag}Total: {total:,}  "
          f"Class 0: {cnt[0]:,} ({100*cnt[0]/total:.1f}%)  "
          f"Class 1: {cnt[1]:,} ({100*cnt[1]/total:.1f}%)  "
          f"Ratio: {cnt[0]/max(cnt[1],1):.1f}:1")
    return dict(cnt)


# ---- Method 1: Random Undersampling ----
def method1_random_undersample(X, y, random_state=42):
    """Randomly downsample majority class to match minority count.
    Constraint: keeps ALL minority class samples; no minority samples removed.
    """
    rng        = np.random.RandomState(random_state)
    idx_1      = np.where(y == 1)[0]
    idx_0      = np.where(y == 0)[0]
    n_min      = len(idx_1)
    selected_0 = rng.choice(idx_0, size=n_min, replace=False)
    combined   = rng.permutation(np.concatenate([idx_1, selected_0]))
    return X[combined], y[combined]


# ---- Method 2: NearMiss-1 Undersampling ----
def method2_nearmiss_undersample(X, y, n_neighbors=3, random_state=42):
    """NearMiss-1: keep n_min majority samples whose average distance to the
    nearest k minority samples is smallest (boundary-aware undersampling).
    Rationale: retains the most informative (hardest) majority samples near the
    decision boundary, improving classifier discrimination.
    Constraint: keeps ALL minority samples; no upsampling performed.
    """
    idx_1 = np.where(y == 1)[0]
    idx_0 = np.where(y == 0)[0]
    n_min = len(idx_1)
    k     = min(n_neighbors, n_min)

    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    knn.fit(X[idx_1])
    dists, _ = knn.kneighbors(X[idx_0])
    mean_dist = dists.mean(axis=1)

    # Keep n_min majority samples closest to minority frontier
    selected_0 = idx_0[np.argsort(mean_dist)[:n_min]]
    combined   = np.random.RandomState(random_state).permutation(
        np.concatenate([idx_1, selected_0]))
    return X[combined], y[combined]


# ---- Inspect original distribution ----
print("Original class distribution:")
check_imbalance_ratio(y_train, "TRAIN")
check_imbalance_ratio(y_test,  "TEST")

# ---- Apply both methods ----
print("\nApplying Method 1: Random Undersampling ...")
X_m1, y_m1 = method1_random_undersample(X_train, y_train, CONFIG["RANDOM_STATE"])
check_imbalance_ratio(y_m1, "Method1-Train")

print("\nApplying Method 2: NearMiss-1 Undersampling ...")
X_m2, y_m2 = method2_nearmiss_undersample(
    X_train, y_train, n_neighbors=3, random_state=CONFIG["RANDOM_STATE"])
check_imbalance_ratio(y_m2, "Method2-Train")

# ---- Verify constraints ----
cnt_orig = Counter(y_train.tolist())
for lbl, y_b in [("Method1", y_m1), ("Method2", y_m2)]:
    cnt_b = Counter(y_b.tolist())
    assert cnt_b[1] == cnt_orig[1], f"{lbl}: minority count changed!"
    assert cnt_b[0] <= cnt_orig[0], f"{lbl}: majority count increased (upsampling detected)!"
print("Constraint checks passed: all minority samples retained, no upsampling.")

# ---- Save datasets as CSV files ----
def save_dataset_csv(X, y, path, feature_names):
    """Save feature matrix + labels to CSV."""
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows  →  {path}")

out = CONFIG["OUTPUT_DIR"]
print("\nSaving datasets ...")
save_dataset_csv(X_train, y_train, out / "imbalanced.csv",                   FEATURE_NAMES)
save_dataset_csv(X_m1,   y_m1,    out / "method1_random_undersample.csv",    FEATURE_NAMES)
save_dataset_csv(X_m2,   y_m2,    out / "method2_nearmiss_undersample.csv",  FEATURE_NAMES)
save_dataset_csv(X_test,  y_test,  out / "test.csv",                          FEATURE_NAMES)

scale_pos_weight_imb = cnt_orig[0] / max(cnt_orig[1], 1)
print(f"\nXGBoost scale_pos_weight (imbalanced dataset): {scale_pos_weight_imb:.2f}")

# In[7]:
# =============================================================
# CELL 7 — Model Configs & Hyperparameter Grids
# =============================================================

def get_model_configs(random_state: int, scale_pos_weight: float) -> list:
    """Return configs for 4 classifiers (2 shallow + 2 ensemble).
    Each config contains: name, label, estimator_fn, param_grids,
                          search_type, n_iter, supports_class_weight.
    """
    cfgs = []

    # 1. Random Forest (Ensemble)
    cfgs.append({
        "name": "RandomForest", "label": "Random Forest",
        "estimator_fn": lambda cw=None: RandomForestClassifier(
            class_weight=cw, random_state=random_state, n_jobs=-1),
        "param_grids": {
            "clf__n_estimators"     : [100, 200, 300],
            "clf__max_depth"        : [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf" : [1, 2, 4],
            "clf__max_features"     : ["sqrt", "log2"],
        },
        "search_type": "random", "n_iter": 15, "supports_class_weight": True,  # Plan A: 30→15
    })

    # 2. SVM — Support Vector Machine (Shallow)
    cfgs.append({
        "name": "SVM", "label": "Support Vector Machine",
        "estimator_fn": lambda cw=None: SVC(
            class_weight=cw, probability=False, random_state=random_state),
        "param_grids": {
            "clf__C"     : [0.1, 1, 10],             # Plan A: tighter grid
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma" : ["scale", "auto"],        # Plan A: tighter grid
        },
        "search_type": "random", "n_iter": 10, "supports_class_weight": True,  # Plan A: 20→10
    })

    # 3. Logistic Regression (Shallow)
    # Two separate param grids avoid illegal solver/penalty combos
    cfgs.append({
        "name": "LogisticRegression", "label": "Logistic Regression",
        "estimator_fn": lambda cw=None: LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=random_state),
        "param_grids": {                           # Plan A: grid→random n_iter=10
            "clf__C"      : [0.01, 0.1, 1, 10, 100],
            "clf__solver" : ["lbfgs", "saga", "liblinear"],
            "clf__penalty": ["l2"],                # avoid solver/penalty conflicts
        },
        "search_type": "random", "n_iter": 10, "supports_class_weight": True,  # Plan A: grid→random
    })

    # 4. XGBoost / GradientBoosting (Ensemble)
    if XGB_AVAILABLE:
        cfgs.append({
            "name": "XGBoost", "label": "XGBoost",
            "estimator_fn": lambda spw=1.0: xgb.XGBClassifier(
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=random_state, n_jobs=-1, verbosity=0),
            "param_grids": {
                "clf__n_estimators"    : [100, 200, 300],
                "clf__max_depth"       : [3, 5, 7],
                "clf__learning_rate"   : [0.05, 0.1, 0.2],
                "clf__subsample"       : [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 1.0],
                "clf__min_child_weight": [1, 3, 5],
            },
            "search_type": "random", "n_iter": 20,  # Plan A: 30→20
            "supports_class_weight": False,
            "scale_pos_weight": scale_pos_weight,
        })
    else:
        cfgs.append({
            "name": "GradientBoosting", "label": "Gradient Boosting",
            "estimator_fn": lambda cw=None: GradientBoostingClassifier(
                random_state=random_state),
            "param_grids": {
                "clf__n_estimators" : [100, 200],
                "clf__max_depth"    : [3, 5, 7],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample"    : [0.7, 0.8, 1.0],
            },
            "search_type": "random", "n_iter": 20, "supports_class_weight": False,
        })

    return cfgs


model_configs = get_model_configs(CONFIG["RANDOM_STATE"], scale_pos_weight_imb)
print("Classifiers configured:")
for mc in model_configs:
    print(f"  [{mc['name']}]  "
          f"search={mc['search_type']}  "
          f"class_weight_support={mc['supports_class_weight']}")

# In[8]:
# =============================================================
# CELL 8 — Training Pipeline (Nested CV + Hyperparameter Tuning)
# =============================================================

SVM_MAX_SAMPLES = 15_000  # SVM is O(n²)–O(n³); cap training size to stay tractable


def build_pipeline(estimator) -> Pipeline:
    """StandardScaler → Classifier pipeline.
    Scaler is fit ONLY on training folds to prevent data leakage.
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def build_search(pipeline, param_grids, search_type, n_iter, inner_cv, rs, n_jobs):
    """Wrap pipeline in GridSearchCV or RandomizedSearchCV."""
    kwargs = dict(cv=inner_cv, scoring="balanced_accuracy",
                  refit=True, n_jobs=n_jobs, error_score=0)
    if search_type == "random":
        return RandomizedSearchCV(pipeline, param_distributions=param_grids,
                                  n_iter=n_iter, random_state=rs, **kwargs)
    return GridSearchCV(pipeline, param_grid=param_grids, **kwargs)


def stratified_subsample(X, y, max_samples, random_state):
    """Stratified subsample of (X, y) down to max_samples rows."""
    rng    = np.random.RandomState(random_state)
    idx_0  = np.where(y == 0)[0]
    idx_1  = np.where(y == 1)[0]
    ratio  = len(idx_1) / len(y)
    n1     = max(1, int(max_samples * ratio))
    n0     = max_samples - n1
    sel_0  = rng.choice(idx_0, size=min(n0, len(idx_0)), replace=False)
    sel_1  = rng.choice(idx_1, size=min(n1, len(idx_1)), replace=False)
    idx    = rng.permutation(np.concatenate([sel_0, sel_1]))
    return X[idx], y[idx]


def run_model(X_tr, y_tr, X_te, y_te, mc, dataset_mode, config) -> dict:
    """Train one classifier on one dataset variant.

    dataset_mode: 'imbalanced' | 'method1' | 'method2'
    For 'imbalanced': class_weight='balanced' (or scale_pos_weight) applied.
    For 'method1'/'method2': data already balanced — no extra weighting needed.

    Training strategy:
      1. Inner 3-fold RandomizedSearch/GridSearch -> best hyperparameters
      2. Outer 5-fold StratifiedKFold -> generalisation metrics (CV)
      3. Refit best estimator on full X_tr -> evaluate on held-out test set
    """
    rs, n_jobs = config["RANDOM_STATE"], config["N_JOBS"]
    is_imb     = (dataset_mode == "imbalanced")

    # SVM: subsample if training set is too large
    if mc["name"] == "SVM" and len(X_tr) > SVM_MAX_SAMPLES:
        X_tr_fit, y_tr_fit = stratified_subsample(X_tr, y_tr, SVM_MAX_SAMPLES, rs)
        print(f"(subsampled {len(X_tr):,}→{len(X_tr_fit):,})", end=" ", flush=True)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    # Build estimator with appropriate imbalance treatment
    if mc["name"] in ("XGBoost", "GradientBoosting"):
        spw = mc.get("scale_pos_weight", 1.0) if is_imb else 1.0
        estimator = mc["estimator_fn"](spw)
    else:
        cw = "balanced" if (is_imb and mc["supports_class_weight"]) else None
        estimator = mc["estimator_fn"](cw)

    # ---- Inner HP search (3-fold) ----
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
    search   = build_search(build_pipeline(estimator), mc["param_grids"],
                            mc["search_type"], mc["n_iter"], inner_cv, rs, n_jobs)
    search.fit(X_tr_fit, y_tr_fit)
    best_pipe   = search.best_estimator_
    best_params = search.best_params_

    # ---- Outer 5-fold CV evaluation ----
    outer_cv = StratifiedKFold(n_splits=config["N_FOLDS"], shuffle=True, random_state=rs)
    cv_y_true, cv_y_pred = [], []
    for tr_idx, val_idx in outer_cv.split(X_tr_fit, y_tr_fit):
        fold_pipe = copy.deepcopy(best_pipe)
        fold_pipe.fit(X_tr_fit[tr_idx], y_tr_fit[tr_idx])
        cv_y_pred.extend(fold_pipe.predict(X_tr_fit[val_idx]).tolist())
        cv_y_true.extend(y_tr_fit[val_idx].tolist())
    cv_metrics = compute_metrics(np.array(cv_y_true), np.array(cv_y_pred))

    # ---- Refit on full X_tr_fit -> test set evaluation ----
    best_pipe.fit(X_tr_fit, y_tr_fit)
    y_pred_test  = best_pipe.predict(X_te)
    test_metrics = compute_metrics(y_te, y_pred_test)

    return {
        "best_params"   : best_params,
        "cv_metrics"    : cv_metrics,
        "test_metrics"  : test_metrics,
        "best_estimator": best_pipe,
        "y_pred_test"   : y_pred_test,
    }


def train_all_models(datasets: dict, X_te, y_te, model_configs, config) -> dict:
    """Train all 4 classifiers on all 3 dataset variants.
    datasets: {'imbalanced': (X_tr, y_tr), 'method1': ..., 'method2': ...}
    Returns nested dict: results[model_name][dataset_mode] = {best_params, ...}
    """
    results = {}
    for mc in model_configs:
        name = mc["name"]
        results[name] = {}
        for dmode, (X_tr, y_tr) in datasets.items():
            print(f"  [{name}] dataset={dmode} ...", end=" ", flush=True)
            res = run_model(X_tr, y_tr, X_te, y_te, mc, dmode, config)
            results[name][dmode] = res
            m = res["test_metrics"]
            print(f"done | bal_acc={m['balanced_accuracy']:.4f}  "
                  f"F1_macro={m['f1_macro']:.4f}  "
                  f"TP={m['TP']}  FN={m['FN']}")
    return results


datasets = {
    "imbalanced": (X_train, y_train),
    "method1"   : (X_m1,   y_m1),
    "method2"   : (X_m2,   y_m2),
}

print("Training all models on 3 dataset variants ...\n")
all_results = train_all_models(datasets, X_test, y_test, model_configs, CONFIG)
print("\nAll models trained.")

# In[9]:
# =============================================================
# CELL 9 — Results Formatting
# =============================================================

DATASET_LABELS = {
    "imbalanced": "No Balancing (Imbalanced)",
    "method1"   : "Method 1: Random Undersampling",
    "method2"   : "Method 2: NearMiss-1 Undersampling",
}


def display_model_report(model_name: str, results: dict) -> None:
    """Print confusion matrix + metrics table for all dataset variants of one model."""
    print("=" * 72)
    print(f"  Model: {model_name}")
    print("=" * 72)
    for dmode, label in DATASET_LABELS.items():
        res = results[dmode]
        print(f"\n--- {label.upper()} ---")
        hps = {k.replace("clf__", ""): v for k, v in res["best_params"].items()}
        print("Best Hyperparameters:", hps)
        print("\nConfusion Matrix (test set):")
        print(format_confusion_table(res["test_metrics"], model_name).to_string())
        print("\nMetrics (test set):")
        print(format_metrics_table(res["test_metrics"]).to_string(index=False))
        print(f"Metrics ({CONFIG['N_FOLDS']}-fold CV on training set):")
        print(format_metrics_table(res["cv_metrics"]).to_string(index=False))


def build_results_summary(all_results: dict) -> pd.DataFrame:
    """Build a wide comparison DataFrame (all models × all dataset variants)."""
    rows = []
    for mname, modes in all_results.items():
        for dmode, res in modes.items():
            m = res["test_metrics"]
            rows.append({
                "Model"            : mname,
                "Dataset"          : DATASET_LABELS[dmode],
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
                "F1 (class 0)"     : m["f1_class_0"],
                "F1 (class 1)"     : m["f1_class_1"],
                "F1 Macro"         : m["f1_macro"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Accuracy"         : m["accuracy"],
            })
    return pd.DataFrame(rows)


for mname, modes in all_results.items():
    display_model_report(mname, modes)
    print()

summary_df = build_results_summary(all_results)
print("\n" + "=" * 72)
print("  COMPARATIVE SUMMARY — TEST SET (All Models × All Dataset Variants)")
print("=" * 72)
try:
    display(summary_df.style
        .format({"F1 (class 0)":":.4f","F1 (class 1)":":.4f",
                 "F1 Macro":":.4f","Balanced Accuracy":":.4f","Accuracy":":.4f"})
        .background_gradient(subset=["Balanced Accuracy","F1 Macro"], cmap="Blues"))
except Exception:
    print(summary_df.to_string(index=False))

# In[10]:
# =============================================================
# CELL 10 — Visualizations
# =============================================================

def plot_confusion_heatmap(cm_dict: dict, title: str, ax) -> None:
    """2×2 confusion matrix heatmap on given axes."""
    matrix = np.array([[cm_dict["TN"], cm_dict["FP"]],
                       [cm_dict["FN"], cm_dict["TP"]]])
    annots = np.array([
        [f"TN\n{cm_dict['TN']}", f"FP\n{cm_dict['FP']}"],
        [f"FN\n{cm_dict['FN']}", f"TP\n{cm_dict['TP']}"],
    ])
    sns.heatmap(matrix, annot=annots, fmt="", cmap="Blues", linewidths=0.5,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Act 0",  "Act 1"],
                cbar=False, ax=ax)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual",    fontsize=8)


def plot_all_confusion_matrices(all_results: dict) -> None:
    """Grid: 4 models (rows) × 3 dataset variants (cols)."""
    model_names = list(all_results.keys())
    dmodes      = list(DATASET_LABELS.keys())
    n_rows, n_cols = len(model_names), len(dmodes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    for r, mname in enumerate(model_names):
        for c, dmode in enumerate(dmodes):
            ax   = axes[r][c] if n_rows > 1 else axes[c]
            cm_d = all_results[mname][dmode]["test_metrics"]
            plot_confusion_heatmap(cm_d, f"{mname}\n{DATASET_LABELS[dmode]}", ax)
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved: confusion_matrices.png")


def plot_all_metric_comparisons(all_results: dict) -> None:
    """2×2 subplot: F1(class1), F1 macro, balanced accuracy, overall accuracy."""
    metric_pairs = [
        ("f1_class_1",         "F1 Score — Class 1 (Fall)"),
        ("f1_macro",           "F1 Macro"),
        ("balanced_accuracy",  "Balanced Accuracy"),
        ("accuracy",           "Overall Accuracy"),
    ]
    model_names = list(all_results.keys())
    dmodes      = list(DATASET_LABELS.keys())
    x      = np.arange(len(model_names))
    width  = 0.25
    colors = ["#5b9bd5", "#ed7d31", "#70ad47"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for (key, label), ax in zip(metric_pairs, axes.flat):
        for i, dmode in enumerate(dmodes):
            vals = [all_results[m][dmode]["test_metrics"][key] for m in model_names]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=DATASET_LABELS[dmode], color=colors[i], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle("Metric Comparison — 3 Dataset Variants (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("metric_comparison.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved: metric_comparison.png")


def plot_feature_importance(model_name: str, best_estimator,
                             feature_names: list, top_n: int = 20) -> None:
    """Horizontal bar chart of top-n feature importances (RF / XGBoost only)."""
    clf = best_estimator.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    imp = clf.feature_importances_
    top = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), imp[top], color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{model_name} — Top {top_n} Features", fontweight="bold")
    plt.tight_layout()
    fname = f"feat_importance_{model_name.lower()}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# Generate all plots
plot_all_confusion_matrices(all_results)
plot_all_metric_comparisons(all_results)
for mc in model_configs:
    name = mc["name"]
    if name in ("RandomForest", "XGBoost", "GradientBoosting"):
        plot_feature_importance(
            name,
            all_results[name]["imbalanced"]["best_estimator"],
            FEATURE_NAMES,
        )

# In[11]:
# =============================================================
# CELL 11 — Final Summary & Best Model Selection
# =============================================================

summary_df = build_results_summary(all_results)

# Per-dataset-variant summary
for dmode, label in DATASET_LABELS.items():
    print(f"\n{'='*65}")
    print(f"  {label}")
    print("="*65)
    sub = (summary_df[summary_df["Dataset"] == label]
           .drop(columns="Dataset")
           .reset_index(drop=True))
    print(sub.to_string(index=False))
    best = sub.loc[sub["Balanced Accuracy"].idxmax()]
    print(f"  >> Best: {best['Model']}  "
          f"Balanced Acc={best['Balanced Accuracy']:.4f}  "
          f"F1 Macro={best['F1 Macro']:.4f}")

# Overall winner
best_row = summary_df.loc[summary_df["Balanced Accuracy"].idxmax()]
print(f"\n{'#'*65}")
print(f"  OVERALL BEST: {best_row['Model']}  on  '{best_row['Dataset']}'")
print(f"  Balanced Accuracy = {best_row['Balanced Accuracy']:.4f}")
print(f"  F1 Macro          = {best_row['F1 Macro']:.4f}")
print(f"  TP={best_row['TP']}  TN={best_row['TN']}  FP={best_row['FP']}  FN={best_row['FN']}")
print("#"*65)
print("\nPipeline complete.")