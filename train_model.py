import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading dataset...")
df = pd.read_csv("synthesis/master_training_dataset.csv")

# Replace missing placeholder
df.replace(-999, np.nan, inplace=True)

# Drop non-numeric identifiers
df = df.drop(columns=["Material_ID", "Formula", "Source_DB"], errors="ignore")

# ==========================================
# CLEAN DATA (robust)
# ==========================================
print("Cleaning data...")
df = df.astype(str).replace(r'[^0-9\.\-]+', '', regex=True)
df = df.apply(pd.to_numeric, errors='coerce')

# ==========================================
# DEFINE FEATURES
# ==========================================
all_properties = [
    "Density_(g/cm3)",
    "Energy_Above_Hull_(eV)",
    "Bulk_Modulus_(GPa)",
    "Shear_Modulus_(GPa)",
    "Ultimate_Tensile_Strength_(MPa)",
    "Yield_Strength_(MPa)",
    "Elongation_(%)",
    "Hardness_(HB)",
    "C_Wt_%",
    "Mn_Wt_%",
    "P_Wt_%",
    "S_Wt_%",
    "Si_Wt_%",
    "Ni_Wt_%",
    "Cr_Wt_%",
    "Mo_Wt_%",
    "Ti_Wt_%"
]

df = df[all_properties]

# ==========================================
# ADD MISSING FLAGS
# ==========================================
for col in all_properties:
    df[col + "_missing"] = df[col].isna().astype(int)

# ==========================================
# IMPUTE
# ==========================================
df.fillna(df.median(), inplace=True)

# ==========================================
# SPLIT
# ==========================================
X = df.copy()
y_raw = df[all_properties]

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=42
)

# ==========================================
# SCALE X
# ==========================================
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# ==========================================
# ✅ SCALE y (CRITICAL FIX)
# ==========================================
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

# ==========================================
# SYNTHETIC MISSINGNESS
# ==========================================
def inject_missingness(X, prob=0.15):
    X = X.copy()
    n = len(all_properties)

    for i in range(X.shape[0]):
        for j in range(n):
            if np.random.rand() < prob:
                X[i, j] = 0.0
                X[i, j + n] = 1
    return X

X_train = inject_missingness(X_train)

# ==========================================
# MODEL
# ==========================================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(len(all_properties))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

print("Training...")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

# ==========================================
# SAVE EVERYTHING
# ==========================================

model.save("matintel_model.h5")
joblib.dump(x_scaler, "scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")
joblib.dump(all_properties, "properties.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Training complete")