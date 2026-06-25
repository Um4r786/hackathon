import numpy as np
import tensorflow as tf
import joblib

print("Loading model...")

# ✅ Fix Keras load issue
model = tf.keras.models.load_model("matintel_model.h5", compile=False)

x_scaler = joblib.load("scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")
properties = joblib.load("properties.pkl")
feature_columns = joblib.load("feature_columns.pkl")

DEFAULT_VALUE = np.nan


# ==========================================
# BUILD INPUT VECTOR
# ==========================================
def build_input_vector(user_inputs):
    vector = []

    for col in feature_columns:

        if col.endswith("_missing"):
            base = col.replace("_missing", "")
            vector.append(0 if base in user_inputs else 1)

        else:
            vector.append(user_inputs.get(col, np.nan))

    x = np.array(vector).reshape(1, -1)

    # Replace NaNs with 0 BEFORE scaling (mean in scaled space)
    x = np.nan_to_num(x, nan=0.0)

    return x


# ==========================================
# CLIP OUTPUT (PHYSICS SAFETY)
# ==========================================
def clip_value(prop, val):
    bounds = {
        "Density_(g/cm3)": (0, 30),
        "Energy_Above_Hull_(eV)": (0, 2),
        "Bulk_Modulus_(GPa)": (0, 500),
        "Shear_Modulus_(GPa)": (0, 300),
        "Ultimate_Tensile_Strength_(MPa)": (0, 2000),
        "Yield_Strength_(MPa)": (0, 1500),
        "Hardness_(HB)": (0, 600),
    }

    if prop in bounds:
        low, high = bounds[prop]
        return float(max(low, min(high, val)))

    return float(val)


# ==========================================
# PREDICT
# ==========================================
def predict(user_inputs):

    x = build_input_vector(user_inputs)
    x_scaled = x_scaler.transform(x)

    preds_scaled = model.predict(x_scaled)
    preds = y_scaler.inverse_transform(preds_scaled)[0]

    results = []

    for i, prop in enumerate(properties):
        results.append({
            "property": prop,
            "value": clip_value(prop, preds[i])
        })

    return results


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    
    test_input = {
        "Density_(g/cm3)": 7.8,
        "C_Wt_%": 0.4,
        "Cr_Wt_%": 1.0,
        "Ni_Wt_%": 0.5,
        "Bulk_Modulus_(GPa)": 160,
    }


    print("\nTest prediction:")
    for r in predict(test_input)[:5]:
        print(r)