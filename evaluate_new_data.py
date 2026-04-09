import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CORE CONFIGURATION
# ==============================================================================
FEATURE_COLS   = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]
LABEL_COL      = "FallCheck"
WINDOW_SIZE    = 50
STEP_SIZE      = 25        
FALL_THRESHOLD = 0.40      
PROJECT_ROOT   = Path(__file__).resolve().parent
MODELS_DIR     = PROJECT_ROOT / "data" / "Balanced" / "saved_models"

STAT_NAMES = ["mean","std","min","max","range","rms","energy",
              "variance","skewness","kurtosis","zcr","mad","p25","p75","iqr"]

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def load_subject_csvs(subject_dir: Path) -> pd.DataFrame:
    required = FEATURE_COLS + [LABEL_COL]
    frames = []
    for csv_path in sorted(subject_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in required)
            if all(c in df.columns for c in required) and len(df) > 0:
                frames.append(df[required].reset_index(drop=True))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required)

def load_dataset(base_dir: Path) -> dict:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory '{base_dir}' not found.")
        sys.exit(1)
    
    subject_data, total_rows, total_files = {}, 0, 0
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        # Maybe the directory directly contains CSV files?
        csvs = list(base_dir.glob("*.csv"))
        if csvs:
            print(f"Loading direct CSVs from {base_dir}...")
            df = load_subject_csvs(base_dir) # Hacking direct folder passing if needed
            if len(df) > 0:
                subject_data["Subject_1"] = df
                total_rows += len(df)
        else:
             print(f"ERROR: No subdirectories or CSV files found in '{base_dir}'.")
             sys.exit(1)
    else:
        for subj_dir in sorted(subdirs):
            total_files += len(list(subj_dir.glob("*.csv")))
            df = load_subject_csvs(subj_dir)
            if len(df) > 0:
                subject_data[subj_dir.name] = df
                total_rows += len(df)
                
    print(f"  Loaded {len(subject_data)} subjects | {total_rows:,} rows of raw data")
    return subject_data

# ==============================================================================
# 3. PREPROCESSING (WINDOWING & EXTRATION)
# ==============================================================================
def apply_sliding_window(df):
    n_rows = len(df)
    if n_rows < WINDOW_SIZE:
        return (np.empty((0, WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float32),
                np.empty((0,), dtype=int))
    feat_arr  = df[FEATURE_COLS].values.astype(np.float32)
    label_arr = df[LABEL_COL].values.astype(int)
    windows, labels = [], []
    for s in range(0, n_rows - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(feat_arr[s : s + WINDOW_SIZE])
        win_lbl = 1 if label_arr[s : s + WINDOW_SIZE].sum() / WINDOW_SIZE > FALL_THRESHOLD else 0
        labels.append(win_lbl)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=int)

def window_dataset(subject_data):
    win_list, lbl_list = [], []
    for subj_id, df in subject_data.items():
        w, l = apply_sliding_window(df)
        if len(w) > 0:
            win_list.append(w); lbl_list.append(l)
    return np.concatenate(win_list), np.concatenate(lbl_list)

def extract_features(windows: np.ndarray) -> np.ndarray:
    W = windows.astype(np.float64)          
    N, T, C = W.shape
    mu      = W.mean(axis=1)                
    sig     = W.std(axis=1)                 
    var     = sig ** 2                      
    mn      = W.min(axis=1)                 
    mx      = W.max(axis=1)                 
    rng     = mx - mn                       
    rms     = np.sqrt((W ** 2).mean(axis=1))
    energy  = (W ** 2).sum(axis=1)          
    p25     = np.percentile(W, 25, axis=1)  
    p75     = np.percentile(W, 75, axis=1)  
    iqr     = p75 - p25                     
    mad     = np.abs(W - mu[:, None, :]).mean(axis=1)  
    
    sig_safe = np.where(sig > 1e-10, sig, 1.0)
    z        = (W - mu[:, None, :]) / sig_safe[:, None, :]  
    skew     = (z ** 3).mean(axis=1)                         
    skew     = np.where(sig > 1e-10, skew, 0.0)
    kurt     = (z ** 4).mean(axis=1) - 3.0                   
    kurt     = np.where(sig > 1e-10, kurt, 0.0)
    
    signs    = np.sign(W)                                     
    zcr      = (np.abs(np.diff(signs, axis=1)).sum(axis=1) / 2 / max(T - 1, 1))                         
    
    X = np.stack([
        mu, sig, mn, mx, rng, rms, energy,
        var, skew, kurt, zcr, mad, p25, p75, iqr,
    ], axis=2)                                                

    return X.reshape(N, -1).astype(np.float32)               

# ==============================================================================
# 4. METRICS & GRADING SYSTEM
# ==============================================================================
def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    EPS = 1e-9
    prec1 = TP / (TP + FP + EPS);  rec1 = TP / (TP + FN + EPS) 
    prec0 = TN / (TN + FN + EPS);  rec0 = TN / (TN + FP + EPS) 
    f1_1  = 2 * prec1 * rec1 / (prec1 + rec1 + EPS)
    f1_0  = 2 * prec0 * rec0 / (prec0 + rec0 + EPS)
    
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision_1": round(prec1, 4), "recall_1": round(rec1, 4),
        "precision_0": round(prec0, 4), "recall_0": round(rec0, 4),
        "f1_class_0": round(f1_0, 4), "f1_class_1": round(f1_1, 4),
        "f1_macro": round((f1_0 + f1_1) / 2, 4),
        "balanced_accuracy": round((rec1 + rec0) / 2, 4),
        "accuracy": round((TP + TN) / max(TP + TN + FP + FN, 1), 4),
    }

# ==============================================================================
# 5. EXECUTION & GRADING FLOW
# ==============================================================================
def print_separator(char="=", length=72): print(char * length)

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_new_data.py <path_to_new_dataset_directory>")
        print("Example: python evaluate_new_data.py data/Sample_Test")
        sys.exit(1)
        
    target_dir = Path(sys.argv[1]).resolve()
    
    print_separator("=")
    print("  FALL DETECTION — AUTOMATED PROFESSOR INFERENCE PIPELINE")
    print_separator("=")
    
    print(f"\n1. Ingesting New Hidden Dataset: {target_dir}")
    raw_data = load_dataset(target_dir)
    
    print("\n2. Processing Target Data (Generating Sliding Windows...)")
    X_windows, y_true = window_dataset(raw_data)
    
    cnt = Counter(y_true.tolist())
    print(f"  -> Generated {len(y_true):,} Test Windows")
    print(f"  -> Ground Truth Distribution — Class 0: {cnt[0]}, Class 1: {cnt[1]}")
    
    print("\n3. Extracting 135 features per window using parallel math algorithms...")
    X_test_extracted = extract_features(X_windows)
    print(f"  -> Successfully generated {X_test_extracted.shape} feature matrix.")

    print("\n4. Loading pre-trained Winning Models from storage...")
    if not MODELS_DIR.exists():
         print(f"[!] Warning: No saved pre-trained models found at {MODELS_DIR}")
         print("[!] Ensure you run 'python main2.py' first until it trains and exports the .pkl models.")
         sys.exit(1)

    model_files = list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        print(f"[!] Warning: No .pkl files found in {MODELS_DIR}")
        sys.exit(1)

    print_separator("-")
    print("                           [ FINAL GRADING REPORT ]")
    print_separator("-")
    
    for mf in model_files:
        model_name = mf.stem
        try:
            model = joblib.load(mf)
            y_pred = model.predict(X_test_extracted)
            metrics = compute_metrics(y_true, y_pred)
            
            print(f"\nEvaluating Model: {model_name.upper()}")
            print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
            print(f"  F1 Macro Score    : {metrics['f1_macro']:.4f}")
            print(f"  F1 (Class 1)      : {metrics['f1_class_1']:.4f}")
            print(f"  Overall Accuracy  : {metrics['accuracy']:.4f}")
            print(f"  Confusion Matrix  : TP={metrics['TP']} | TN={metrics['TN']} | FP={metrics['FP']} | FN={metrics['FN']}")
        except Exception as e:
            print(f"  [Error loading or predicting with {model_name}]: {e}")
            
    try:
        # Optional: Attempt Keras
        keras_models = list(MODELS_DIR.glob("*.keras"))
        from tensorflow import keras
        for pf in keras_models:
             model = keras.models.load_model(pf)
             y_prob = model.predict(X_windows, verbose=0).ravel()
             y_pred = (y_prob >= 0.5).astype(int)
             metrics = compute_metrics(y_true, y_pred)
             
             print(f"\nEvaluating Model: {pf.stem.upper()}")
             print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
             print(f"  F1 Macro Score    : {metrics['f1_macro']:.4f}")
             print(f"  F1 (Class 1)      : {metrics['f1_class_1']:.4f}")
             print(f"  Overall Accuracy  : {metrics['accuracy']:.4f}")
             print(f"  Confusion Matrix  : TP={metrics['TP']} | TN={metrics['TN']} | FP={metrics['FP']} | FN={metrics['FN']}")
    except Exception:
        pass # Ignore Keras logic if TF missing on Professor's machine

    print("\n" + "="*72)
    print("  INFERENCE COMPLETE.")
    print("="*72)

if __name__ == "__main__":
    main()
