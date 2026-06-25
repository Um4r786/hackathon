import pandas as pd
import numpy as np
 
# Load dataset
df = pd.read_csv("synthesis/master_training_dataset.csv")
 
# Replace -999 with NaN
df.replace(-999, np.nan, inplace=True)
 
 
# ----------------------------------------
# MULTI-PROPERTY SIMILARITY FUNCTION
# ----------------------------------------
def get_top3_similar(material_id, input_properties, target_property):
   
    """
    input_properties = dictionary of known values
    Example:
    {
        "Density_(g/cm3)": 6.75,
        "Energy_Above_Hull_(eV)": 0.18
    }
    """
 
    # Keep only rows where all required properties exist
    required_cols = list(input_properties.keys()) + [target_property]
    df_clean = df.dropna(subset=required_cols).copy()
 
    # Compute std for each property
    stds = {}
    for prop in input_properties:
        stds[prop] = df_clean[prop].std()
 
    # Apply ±2σ condition for ALL properties
    mask = np.ones(len(df_clean), dtype=bool)
 
    for prop, value in input_properties.items():
        lower = value - 2 * stds[prop]
        upper = value + 2 * stds[prop]
       
        mask &= df_clean[prop].between(lower, upper)
 
    similar = df_clean[mask].copy()
 
    # Remove input material
    similar = similar[similar["Material_ID"] != material_id]
 
    # Rank by total distance across all properties
    similar["total_distance"] = 0
 
    for prop, value in input_properties.items():
        similar["total_distance"] += abs(similar[prop] - value)
 
    similar = similar.sort_values(by="total_distance")
 
    # Return top 3
    return similar.head(3)[["Material_ID", "Formula", target_property]]
 
 
 
result = get_top3_similar(
    material_id="mp-1521506",
    input_properties={
        "Density_(g/cm3)": 6.75,
        "Energy_Above_Hull_(eV)": 0.23
    },
    target_property="Energy_Above_Hull_(eV)"
)
 
print(result)