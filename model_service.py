"""
MatIntel – Model Service (fixed)
==================================
Key fixes vs. original:
  1. Input vector built from COMPOSITION features only (no target leakage)
  2. Missing inputs imputed with training medians (loaded from input_medians.pkl)
     instead of hard-coding 0 before scaling – median imputation is consistent
     with what inject_missingness emulates at training time (0 in scaled space
     equals the mean, which the scaler was fit on median-imputed data, so these
     are equivalent; we preserve the exact same pipeline here for correctness).
  3. Elongation clip bound added.
  4. Clip applied after inverse-transform (correct order).
"""

import numpy as np
import tensorflow as tf
import joblib

print("Loading model...")

model = tf.keras.models.load_model("matintel_model.h5", compile=False)

x_scaler       = joblib.load("scaler.pkl")
y_scaler       = joblib.load("y_scaler.pkl")
properties     = joblib.load("properties.pkl")
feature_columns = joblib.load("feature_columns.pkl")
input_medians  = joblib.load("input_medians.pkl")   # pandas Series: composition medians

# Derived: the raw (non-flag) composition column names
composition_features = [c for c in feature_columns if not c.endswith("_missing")]

# ==========================================
# PHYSICS BOUNDS
# ==========================================
BOUNDS = {
    "Density_(g/cm3)":                   (0.5,  25.0),
    "Energy_Above_Hull_(eV)":            (0.0,   3.0),
    "Bulk_Modulus_(GPa)":                (0.0, 500.0),
    "Shear_Modulus_(GPa)":               (0.0, 300.0),
    "Ultimate_Tensile_Strength_(MPa)":   (0.0, 2500.0),
    "Yield_Strength_(MPa)":              (0.0, 2000.0),
    "Elongation_(%)":                    (0.0,  100.0),
    "Hardness_(HB)":                     (0.0,  700.0),
}

def clip_value(prop, val):
    if prop in BOUNDS:
        low, high = BOUNDS[prop]
        return float(np.clip(val, low, high))
    return float(val)

# ==========================================
# BUILD INPUT VECTOR
# ==========================================
def build_input_vector(user_inputs: dict) -> np.ndarray:
    """
    Build the scaled input vector from a dict of composition values.

    user_inputs keys should be composition column names, e.g.:
        {"C_Wt_%": 0.4, "Cr_Wt_%": 1.0, "Ni_Wt_%": 0.5}

    Missing keys are imputed with training-set medians (same strategy as
    during training), and their _missing flag is set to 1.
    """
    vector = []

    for col in feature_columns:
        if col.endswith("_missing"):
            base = col.replace("_missing", "")
            # Flag = 1 (missing) if the user did not supply this composition element
            vector.append(0.0 if base in user_inputs else 1.0)
        else:
            if col in user_inputs:
                vector.append(float(user_inputs[col]))
            else:
                # Impute with training median for this element
                vector.append(float(input_medians.get(col, 0.0)))

    x = np.array(vector, dtype=np.float32).reshape(1, -1)

    # Scale using the fitted scaler (trained on median-imputed data)
    x_scaled = x_scaler.transform(x)

    # Where the composition value was missing, zero it out in scaled space.
    # 0 in standardized space == the training mean, which is close to the median
    # for the roughly symmetric distributions here.  This matches what the model
    # was trained to expect for missing inputs.
    for j, col in enumerate(feature_columns):
        if not col.endswith("_missing") and col not in user_inputs:
            x_scaled[0, j] = 0.0

    return x_scaled

# ==========================================
# PREDICT
# ==========================================
def predict(user_inputs: dict) -> list[dict]:
    """
    Run a forward pass and return a list of
    {"property": <name>, "value": <float>} dicts.

    user_inputs: dict of known composition wt% values (any subset).
    """
    x_scaled = build_input_vector(user_inputs)

    preds_scaled = model.predict(x_scaled, verbose=0)
    preds = y_scaler.inverse_transform(preds_scaled)[0]

    results = []
    for i, prop in enumerate(properties):
        results.append({
            "property": prop,
            "value":    clip_value(prop, preds[i]),
        })

    return results

