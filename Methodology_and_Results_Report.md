# Fall Detection Pipeline Methodology & Comparative Analysis

## 1. Objective
This project develops an advanced machine learning pipeline capable of differentiating between "Fall" and "No-Fall" states utilizing human-mounted spatial sensors (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ) processing at 100 Hz.

A critical challenge identified in raw IMU datasets is **extreme class imbalance**, where routine activity (No-Fall) massively outnumbers critical anomaly events (Falls).

As specifically instructed, this report contrasts pipeline performance across **three rigorous distribution controls** to illustrate how balancing corrects the minority blind spot without corrupting genuine anomaly distributions:
1.  **Purely Imbalanced Baseline:** Ground truth sensor data with absolutely no algorithmic or data-level intervention.
2.  **Algorithmically Weighted Baseline:** The data remains physically untouched, but algorithms mathematically penalize errors on minority samples to perfectly equalize geometric loss surfaces.
3.  **Physically Balanced Data (Minority Constraint):** The dataset is forced to a 1:1 state constraint where the majority class is randomly undersampled, but **100% of all minority (Fall) samples are stringently retained** to avoid throwing out ground-truth positive cases.

---

## 2. Preprocessing & Feature Engineering

### 2.1 Sliding Window Segmentation
Streaming 100 Hz signal data point-by-point lacks temporal context. We segment the streams into a sliding window of **50 timesteps** (0.5 seconds of context per window), advancing by a step size of **25 timesteps** (50% overlap).
*   **Label Thresholding:** If more than 40% of the discrete points within a 50-step window indicate a fall, the entire aggregated block is classified as a Positive (1) Fall instance. Otherwise, it is a Routine (0) Negative instance.

### 2.2 Vectorised Statistical Extraction
A neural network LSTM block processes raw 3D continuous data directly as a baseline. For standard classification models (Random Forest, SVM, Logistic Regression, XGBoost), feeding flat consecutive waves is detrimental. Instead, we project the 50-step timeline into a highly compressed statistical vector space extracting **15 core measurements** per channel:
*   *Central Tendencies & Variance:* Mean, Std, Variance, Median, Interquartile Range, p25, p75.
*   *Energy & Displacements:* Min, Max, Range, Root Mean Square (RMS), Energy.
*   *Wave Shape:* Skewness, Excess Kurtosis, Zero-Crossing Rate.

15 measurements × 9 sensor channels yield exactly **135 Features per block**. This shrinks the dimensional space footprint and provides mathematically descriptive markers of sudden velocity or impact states.

---

## 3. Dataset Variants and Sampling Strategy

Our raw sliding window procedure produces roughly 49,000 blocks for training.
*   **Total Windows:** 49,290
*   **No-Fall (0):** 43,160  (87.6%)
*   **Fall (1):** 6,130  (12.4%)
*   **Ratio:** 7.04 to 1

To explicitly compare model behavior when exposed to balanced logic against naive training logic, three pathways were designed:

### Variant 1: Purely Imbalanced Model (Baseline)
Models receive the 7:1 dataset exactly as is. **No class weights** or algorithmic assistance scale the gradients during training. This simulates the standard textbook machine learning flaw: the algorithm realizes it attains 87% test accuracy just by blindly guessing "No-Fall" to everything, obliterating Recall.

### Variant 2: Algorithmically Balanced (Class-Weighted)
The dataset structure stays 49,290 windows strong. However, models receive a calculated class-weight parameter map (`scale_pos_weight = 7.04`). The loss function is inverted so that failing to predict a Fall incurs a 704% heavier mathematical penalty. The model is forced to acknowledge the critical, minority data without any physical data being deleted.

### Variant 3: Physically Data-Balanced (Random Undersampling the Majority)
The dataset is physically chopped exactly in half to enforce a 1:1 mapping through random undersampling.
*   **Crucial Architectural Rule Applied:** Undersampling techniques notoriously risk throwing away critical Positive class information. Our function is explicitly hard-coded: **Leave 100% of the minority class untouched.**
*   All 6,130 Fall samples are retained. We extract exactly 6,130 random samples from the 43,160 No-Fall bucket. This guarantees the model trains on the total variation of actual Falls possible.

> [!NOTE]
> *Note on SVM Constraints: Because traditional Support Vector Machines calculate point boundaries at O(n²) complexity, attempting to feed 49,000 windows causes a memory and runtime stall. The SVM specifically handles memory limits through a similar constraint down to 15,000 limits—where it prioritizes keeping 100% of minority points and capping the remaining majority rows to prevent freezing.*

---

## 4. Hyperparameter Search & Deep Validation

To fairly benchmark, all four classical models are run through a strictly robust Nested Cross-Validation harness.
1.  **Inner Validation (Grid Search):** A randomized grid search tests 15 or 20 combinations of settings (e.g. tree depth, learning rate) applying a 3-Fold Stratified Split. The search optimizing mechanism is aggressively tuned to chase **Balanced Accuracy**—not generic accuracy.
2.  **Outer Evaluation:** The optimal settings found via GridSearch are cross-validated through 3 external folds to track their convergence, and then finally evaluated against the unseen ~11,000 Test Set windows.

## 5. Metrics & Rationale
We ignore standard `Accuracy` as the main metric. An algorithm predicting 0% of falls could achieve 87% standard Accuracy on this specific test set.
Instead, we pivot entirely to evaluating:
*   **Balanced Accuracy:** The macroscopic average of True Positive Rate (Sensitivity) and True Negative Rate.
*   **F1-Macro / Class-1 F1:** The harmonic mean between Precision and Recall. F1 strictly guarantees the models are neither crying wolf on false positives nor ignoring actual positive impacts.
