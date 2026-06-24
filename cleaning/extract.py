import pandas as pd
import numpy as np

# 1. Load the data (Using the 5000 row trick for Alexandria!)
alexandria_df = pd.read_csv("alexandria_sample_5000.csv", nrows=100)
stainless_df = pd.read_csv("stainless_steel.csv", encoding='latin1')
carbon_df = pd.read_csv("carbon_steel.csv", encoding='latin1')

# 2. Map Alexandria
alexandria_mapped = alexandria_df.rename(columns={
    'formula': 'material_id',
    'band_gap_dir': 'band_gap',
    'energy_total': 'total_energy',
    'volume': 'volume'
})

# 3. ONE universal dictionary for both Steel datasets!
steel_mapping = {
    'SAE Grade': 'material_id',
    'UTS (MPa)': 'ultimate_tensile_strength_mpa',
    'YS (MPa)': 'yield_strength_mpa', 
    'Elongation (%)': 'elongation_percent',
    
    # Using the Max values to train the ML model
    'C (Max)': 'carbon',
    'Mn (Max)': 'manganese',
    'Cr (Max)': 'chromium',
    'Ni (Max)': 'nickel',
    'Si (Max)': 'silicon',
    'Mo (Max)': 'molybdenum'
}

stainless_mapped = stainless_df.rename(columns=steel_mapping)
carbon_mapped = carbon_df.rename(columns=steel_mapping)

# Add our source tracking flags
alexandria_mapped['source'] = 'Alexandria'
stainless_mapped['source'] = 'Stainless'
carbon_mapped['source'] = 'Carbon'

# 4. Merge them together and fix the Keras NaN issue
matintel_master = pd.concat([alexandria_mapped, stainless_mapped, carbon_mapped], ignore_index=True)
matintel_master = matintel_master.fillna(-1)

# 5. Define the Final Features (X) and Target (y)
features = [
    'carbon', 'manganese', 'chromium', 'nickel', 'silicon', 'molybdenum', 
    'band_gap', 'total_energy', 'volume'
]
target = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa', 'elongation_percent']

# Extract the arrays for TensorFlow
X = matintel_master[features].values
y = matintel_master[target].values

print(f"X shape (Features): {X.shape}")
print(f"y shape (Target): {y.shape}")

# Save the final file so you can use it in your Streamlit app
matintel_master.to_csv("MatIntel_Master_Final.csv", index=False)