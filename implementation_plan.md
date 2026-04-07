# Speed Overhaul — Fall Detection Pipeline

The current `main.ipynb` hangs because three compounding bottlenecks make Cell 8 run for **hours**:

| Bottleneck | Root Cause |
|---|---|
| Feature extraction (Cell 4) | Python `for w in windows` loop over 49 k windows |
| Model search (Cell 8) | `RandomizedSearchCV(n_iter=30, cv=3)` × 3 datasets × 4 models |
| GradientBoosting | sklearn's slow, single-threaded implementation |
| SVM outer 5-fold CV | Even the subsampled 15k set is O(n²) inside 5 folds |

## Proposed Changes

### Plan A — "Fast-ML" (still classical, DL-free)

Keep the same academic story (sliding window → hand-crafted features → classifiers), but apply surgical speedups:

---

#### [MODIFY] [main.ipynb](file:///c:/Users/BIT/Documents/Super_Coding/Projects/DataMiners/notebooks/projNbs/main.ipynb)

**Cell 1 – Imports**
- Add `xgboost` (install via `pip install xgboost`; it is GPU-aware and parallelised)
- Remove `GradientBoostingClassifier` import (replaced by XGBoost)

**Cell 4 – Feature extraction (vectorised)**
Replace the per-window Python loop with a **fully vectorised NumPy approach**:
```python
# Instead of: [compute_window_features(w) for w in windows]
# Use:        axis-wise reductions on the 3-D array
X = windows  # shape (N, 50, 9)
mean   = X.mean(axis=1)        # (N, 9)
std    = X.std(axis=1)
# ... all 15 stats in one shot
```
This is **40–80× faster** than the Python loop (~1 s vs ~60 s for 49 k windows).

**Cell 7 – Model configs**
| Old | New | Why |
|---|---|---|
| `GradientBoostingClassifier(n_iter 20, random)` | `XGBClassifier` | 5–20× faster, GPU support |
| `SVC` random search n_iter=20, outer 5-fold | `SVC` n_iter=10, outer 3-fold, tighter C/gamma grid | Halves SVM search time |
| RF random n_iter=30 | RF n_iter=15 | Enough for academic purposes |
| LR grid (12×2 combos) | LR random n_iter=10 | Cuts grid search |
| inner_cv 3-fold everywhere | remains 3-fold | Already minimal |
| outer_cv 5-fold | **3-fold** (still statistically valid) | 40% less CV time |

> [!IMPORTANT]
> The academic structure (sliding window → hand-crafted features → 4 classifiers × 3 datasets) stays **100% intact**. Only iteration counts and implementation efficiency change.

---

### Plan B — "DL Fast" (bonus cell, LSTM)

Add an **optional Cell 8b** with a `keras`/`torch` 1-D CNN or LSTM that:
- Takes the **raw windows** (N, 50, 9) directly (no feature extraction step needed)  
- Trains in ~30 s on CPU with `epochs=10, batch_size=256`
- Provides a DL baseline to compare against the classical models

> [!NOTE]
> Plan B is additive — it doesn't replace any existing cells. Recommend this only if the professor expects a DL comparison.

---

## Estimated Speedup

| Step | Before | After (Plan A) |
|---|---|---|
| Feature extraction | ~60–90 s | ~1–2 s |
| RF (×3 datasets) | 15 min+ | ~2–3 min |
| SVM (×3 datasets) | 20 min+ | ~4–5 min |
| LR (×3 datasets) | 5 min+ | ~1 min |
| XGBoost (×3 datasets, replaces GB) | 10 min for GB | ~1–2 min |
| **Total** | **50 min – 3 h** | **~10–12 min** |

## Open Questions

> [!IMPORTANT]
> **Do you want Plan B (DL cell) added?**
> A simple 1D-CNN or LSTM adds ~30 extra seconds of training and gives you a nice DL vs classical comparison table. Let me know if you want this included.

> [!NOTE]
> **XGBoost already installed?** If not, I'll add a `!pip install xgboost` guard at the top of Cell 1. Let me know if you have it.

## Verification Plan

### Automated
- Re-run all 11 cells top-to-bottom in Jupyter
- Cell 8 should complete in < 15 minutes locally

### Output Parity Check
- Summary table (Cell 9) should produce metrics similar to the existing saved outputs (RF bal_acc ≈ 0.95, SVM bal_acc ≈ 0.95)
- All plots from Cell 10 should render without error
