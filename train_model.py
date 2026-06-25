"""
MatIntel – Model Training Script (fixed)
=========================================
Key fixes vs. original:
  1. Sentinel handling: replace -999 AFTER to_numeric (Hardness was str dtype)
  2. Data leakage removed: X uses only COMPOSITION inputs; targets are PROPERTY outputs
  3. inject_missingness operates correctly on the input-feature matrix
  4. Elongation clip bound added to model_service
  5. y_scaler is fitted only on training split, not full dataset
  6. Sparse data handled: Steel DB (mechanical) and MP DB (physical) trained separately
     via masked loss so each row only contributes to the properties it actually has.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# LOAD & CLEAN
# ==========================================
print("Loading dataset...")
df = pd.read_csv("synthesis/master_training_dataset.csv")

# Drop non-numeric identifiers
df = df.drop(columns=["Material_ID", "Formula", "Source_DB"], errors="ignore")

# 1. Strip stray non-numeric characters first, THEN convert, THEN replace sentinel
#    (Hardness column is object/str dtype in the raw file – replace(-999) on int/str
#    before coercion has no effect on the string column.)
print("Cleaning data...")
df = df.astype(str).replace(r'[^0-9\.\-eE]+', '', regex=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Replace the -999 sentinel NOW that all columns are numeric
df.replace(-999, np.nan, inplace=True)

# ==========================================
# SPLIT INTO INPUT FEATURES AND TARGETS
# ==========================================
# Composition columns – what an engineer knows about a candidate material.
# These are the ONLY model inputs to avoid data leakage (targets must not
# appear in the feature matrix).
composition_features = [
    "C_Wt_%",
    "Mn_Wt_%",
    "P_Wt_%",
    "S_Wt_%",
    "Si_Wt_%",
    "Ni_Wt_%",
    "Cr_Wt_%",
    "Mo_Wt_%",
    "Ti_Wt_%",
]

# Material properties to predict
target_properties = [
    "Density_(g/cm3)",
    "Energy_Above_Hull_(eV)",
    "Bulk_Modulus_(GPa)",
    "Shear_Modulus_(GPa)",
    "Ultimate_Tensile_Strength_(MPa)",
    "Yield_Strength_(MPa)",
    "Elongation_(%)",
    "Hardness_(HB)",
]

# ==========================================
# ADD MISSING FLAGS (inputs only)
# ==========================================
# For each composition feature add a binary flag: 1 = value was missing.
# This teaches the model which inputs were actually provided vs. imputed.
X_raw = df[composition_features].copy()
for col in composition_features:
    X_raw[col + "_missing"] = X_raw[col].isna().astype(float)

y_raw = df[target_properties].copy()

feature_columns = X_raw.columns.tolist()
n_features = len(composition_features)   # number of raw (non-flag) feature columns

# ==========================================
# DROP ROWS THAT HAVE NO TARGET AT ALL
# ==========================================
# A row with every target missing gives the model nothing to learn from.
valid_mask = y_raw.notna().any(axis=1)
X_raw = X_raw[valid_mask].reset_index(drop=True)
y_raw = y_raw[valid_mask].reset_index(drop=True)
print(f"Usable rows after filtering: {len(X_raw)}")

# ==========================================
# IMPUTE MISSING INPUTS WITH COLUMN MEDIAN
# ==========================================
input_medians = X_raw[composition_features].median()
for col in composition_features:
    X_raw[col] = X_raw[col].fillna(input_medians[col])

# ==========================================
# TRAIN / TEST SPLIT  (before fitting scalers)
# ==========================================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# ==========================================
# SCALE INPUTS
# ==========================================
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train_raw)   # fit on train only
X_test  = x_scaler.transform(X_test_raw)

# ==========================================
# SYNTHETIC MISSINGNESS (augmentation)
# Randomly zero out composition values and flip their flag to 1.
# Only operates on the n_features raw-value columns; flag columns are index n..2n.
# ==========================================
def inject_missingness(X, prob=0.15):
    X = X.copy()
    for i in range(X.shape[0]):
        for j in range(n_features):            # iterate over composition cols only
            if np.random.rand() < prob:
                X[i, j] = 0.0                  # 0 in scaled space == mean (correct)
                X[i, j + n_features] = 1.0     # flag column (correctly offset by n_features)
    return X

X_train = inject_missingness(X_train)

# ==========================================
# SCALE TARGETS  (fit on train split only)
# ==========================================
# Impute NaN targets with per-column training median before scaling so that
# StandardScaler.fit() has no NaNs.  We keep a separate mask to implement
# masked loss (see below) so imputed values never contribute to the gradient.
y_train_median = y_train_raw.median()
y_train_imputed = y_train_raw.fillna(y_train_median)
y_test_imputed  = y_test_raw.fillna(y_train_median)   # use TRAIN medians for test too

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train_imputed)
y_test  = y_scaler.transform(y_test_imputed)

# Binary masks: 1 where the target was genuinely observed, 0 where it was missing
train_mask = (~y_train_raw.isna()).values.astype(np.float32)
test_mask  = (~y_test_raw.isna()).values.astype(np.float32)

n_outputs = len(target_properties)

# ==========================================
# MASKED LOSS
# Rows from the MP database have no mechanical properties; rows from the
# steel database have no modulus / density.  A vanilla MSE would push the
# model to predict training medians for every missing cell, biasing it.
# Masked MSE only back-propagates through cells that were actually observed.
# ==========================================
def masked_mse(y_true, y_pred):
    # y_true is packed as [scaled_targets | observed_mask] along the last axis
    half = tf.shape(y_true)[1] // 2
    targets = y_true[:, :half]
    mask    = y_true[:, half:]
    squared_err = tf.square(targets - y_pred)
    masked_err  = squared_err * mask
    # Normalise by the number of observed cells (avoid divide-by-zero)
    n_observed = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(masked_err) / n_observed

# Pack targets and masks together for Keras
y_train_packed = np.concatenate([y_train, train_mask], axis=1)
y_test_packed  = np.concatenate([y_test,  test_mask],  axis=1)

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
n_input_cols = X_train.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(n_input_cols,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(n_outputs),   # one output per target property
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=masked_mse,
)

model.summary()

# ==========================================
# TRAINING
# ==========================================
print("Training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
]

history = model.fit(
    X_train, y_train_packed,
    validation_data=(X_test, y_test_packed),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ==========================================
# SAVE ARTIFACTS
# ==========================================
model.save("matintel_model.h5")
joblib.dump(x_scaler,          "scaler.pkl")
joblib.dump(y_scaler,          "y_scaler.pkl")
joblib.dump(target_properties, "properties.pkl")
joblib.dump(feature_columns,   "feature_columns.pkl")
joblib.dump(input_medians,     "input_medians.pkl")   # needed by model_service for imputation

print("✅ Training complete")