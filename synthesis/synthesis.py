import pandas as pd
import numpy as np
from mp_api.client import MPRester

# ==========================================
# 1. MATERIALS PROJECT EXTRACTION
# ==========================================
def extract_mp_data(api_key):
    print("Connecting to Materials Project API...")
    desired_fields = [
        "material_id", "formula_pretty", "density",
        "energy_above_hull", "bulk_modulus", "shear_modulus"
    ]
    
    data = []
    # Broaden structural metals, but we will search them ONE at a time (OR logic)
    structural_metals = ["Fe", "Al", "Ti", "Ni", "Mg", "Cu"]
    
    with MPRester(api_key) as mpr:
        for metal in structural_metals:
            print(f"Fetching {metal} materials with elasticity data...")
            docs = mpr.materials.summary.search(
                elements=[metal],
                has_props=["elasticity"], # ONLY get rows with Bulk/Shear Modulus
                fields=desired_fields
            )
            
            for doc in docs:
                # Safely extract the VRH average
                b_mod = doc.bulk_modulus.get('vrh') if isinstance(doc.bulk_modulus, dict) else doc.bulk_modulus
                s_mod = doc.shear_modulus.get('vrh') if isinstance(doc.shear_modulus, dict) else doc.shear_modulus
                
                data.append({
                    "Material_ID": str(doc.material_id),
                    "Formula": doc.formula_pretty,
                    "Density_(g/cm3)": doc.density,
                    "Energy_Above_Hull_(eV)": doc.energy_above_hull,
                    "Bulk_Modulus_(GPa)": b_mod,
                    "Shear_Modulus_(GPa)": s_mod,
                    "Source_DB": "Materials_Project"
                })
                
    # Convert to DataFrame
    mp_df = pd.DataFrame(data)
    
    # CRITICAL: Drop duplicates! (e.g. an Fe-Al alloy will be pulled in both the Fe and Al loops)
    mp_df = mp_df.drop_duplicates(subset=["Material_ID"])
    
    print(f"Extracted {len(mp_df)} unique materials from MP.")
    return mp_df


# ==========================================
# 2. LOCAL KAGGLE STEEL DATA PROCESSING
# ==========================================
def process_local_steel_data(carbon_path, stainless_path):
    print("Processing local steel CSVs...")
    
    # Load files (handling potential encoding issues)
    try:
        carbon = pd.read_csv(carbon_path, encoding='utf-8')
    except UnicodeDecodeError:
        carbon = pd.read_csv(carbon_path, encoding='latin-1')
        
    try:
        stainless = pd.read_csv(stainless_path, encoding='utf-8')
    except UnicodeDecodeError:
        stainless = pd.read_csv(stainless_path, encoding='latin-1')
        
    # Combine Carbon and Stainless into one DataFrame
    steel_df = pd.concat([carbon, stainless], ignore_index=True)
    
    # Drop rows that don't actually have mechanical data (empty header rows)
    steel_df = steel_df.dropna(subset=['UTS (MPa)', 'YS (MPa)'], how='all')
    
    # Create standardized identifiers
    steel_df['Material_ID'] = steel_df['SAE Grade'].astype(str) + " (" + steel_df['Conditions'].astype(str) + ")"
    steel_df['Formula'] = "Fe-Alloy"
    steel_df['Source_DB'] = "Local_Steel_CSV"
    
    # Standardize our Target Property Names
    steel_df = steel_df.rename(columns={
        'UTS (MPa)': 'Ultimate_Tensile_Strength_(MPa)',
        'YS (MPa)': 'Yield_Strength_(MPa)',
        'Elongation (%)': 'Elongation_(%)',
        'Hardness (HB)': 'Hardness_(HB)'
    })
    
    # FEATURE ENGINEERING: Convert Min/Max ranges into a single average weight percentage
    elements = ['C', 'Mn', 'P', 'S', 'Si', 'Ni', 'Cr', 'Mo', 'Ti']
    for el in elements:
        min_col = f'{el} (Min)'
        max_col = 'S(Max)' if el == 'S' else f'{el} (Max)' 
        
        if min_col in steel_df.columns and max_col in steel_df.columns:
            # 1. FORCE THE COLUMNS TO BE NUMERIC (The Fix)
            # errors='coerce' turns string garbage like "<0.08" into NaN
            steel_df[min_col] = pd.to_numeric(steel_df[min_col], errors='coerce')
            steel_df[max_col] = pd.to_numeric(steel_df[max_col], errors='coerce')
            
            # 2. Now calculate the mean safely
            steel_df[f'{el}_Wt_%'] = steel_df[[min_col, max_col]].mean(axis=1)
            
    # Keep only the cleaned columns
    cols_to_keep = [
        'Material_ID', 'Formula', 'Source_DB', 
        'Ultimate_Tensile_Strength_(MPa)', 'Yield_Strength_(MPa)', 
        'Elongation_(%)', 'Hardness_(HB)'
    ] + [f'{el}_Wt_%' for el in elements]
    
    # Filter to only keep columns that survived the cleaning process
    cols_to_keep = [c for c in cols_to_keep if c in steel_df.columns]
    
    print(f"Processed {len(steel_df)} local steel records.")
    return steel_df[cols_to_keep]


# ==========================================
# 3. SYNTHESIS AND EXECUTION
# ==========================================
if __name__ == "__main__":
    # Insert your actual MP API Key here
    YOUR_API_KEY = "ShT9Y6gEcms8gYORcn2bzrmMBRxDoZec"    
    # 1. Fetch MP Data
    mp_df = extract_mp_data(YOUR_API_KEY)
    
    # 2. Process Local CSVs
    steel_df = process_local_steel_data(
        "carbon_steel.csv", 
        "stainless_steel.csv"
    )
    
    # 3. SYNTHESIZE: Concatenate them into a single master sheet
    print("Synthesizing datasets...")
    master_training_data = pd.concat([mp_df, steel_df], ignore_index=True)
    master_training_data = master_training_data.fillna(-999)
    
    # 4. Save the Final Output
    master_training_data.to_csv("master_training_dataset.csv", index=False)
    print("Success! Master training dataset saved as 'master_training_dataset.csv'.")
    print(f"Total Materials in Database: {len(master_training_data)}")