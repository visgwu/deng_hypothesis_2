import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURATION ---
TRAIN_FILE = 'data/train_dataset.csv'

# Your requested 10 features map to these columns in the CSV
FEATURE_MAP = {
    'builder_id': 'builder.id',
    'recipe_type': 'recipe.type',
    'recipe_entry_point': 'recipe.entryPoint',
    'recipe_defined_in_material': 'recipe.definedInMaterial',
    'material_uri': 'materials[*].uri',
    'material_digest': 'materials[*].digest',
    'recipe_arguments': 'recipe.arguments',
    'recipe_environment': 'recipe.environment', # Synthetic column we add
    'metadata_completeness_materials': 'metadata.completeness.materials',
    'metadata_reproducible': 'metadata.reproducible'
}

def analyze():
    print("1. Loading Real-World Metadata...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    # Load Clean Data
    df_clean = pd.read_csv(TRAIN_FILE)
    df_clean['label'] = 0
    df_clean['recipe_environment'] = "{}" # Fill missing feature
    
    # 2. Synthesize Tampered Data (Label 1)
    print("2. Synthesizing Tampered Examples for Contrast...")
    df_bad = df_clean.copy()
    df_bad['label'] = 1
    
    # Inject anomalies into specific FIELDS to see if model catches them
    # We change the FIELD content, not just words
    df_bad['builder_id'] = "https://untrusted-builder.com/executor"
    df_bad['recipe_entry_point'] = "malicious_install.sh"
    df_bad['recipe_environment'] = '{"LD_PRELOAD": "/tmp/malware.so"}'
    
    # Combine
    df_final = pd.concat([df_clean, df_bad], ignore_index=True)
    
    # 3. Feature Engineering (Column-based)
    print("3. extracting features (Column-Level)...")
    
    X = pd.DataFrame()
    
    # We encode each column as a single feature (Number)
    # This ensures the importance score is assigned to the Column, not the word 'http'
    for col_name, feature_name in FEATURE_MAP.items():
        if col_name in df_final.columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types/nulls
            X[feature_name] = le.fit_transform(df_final[col_name].astype(str))
        else:
            print(f"Warning: Column {col_name} not found in CSV.")

    y = df_final['label']
    
    # 4. Train Model
    print("4. Training Analysis Model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 5. Extract Importance
    importances = rf.feature_importances_
    
    res = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    res = res.sort_values('Importance', ascending=False)
    
    print("\n--- Feature Importance (Requested 10 Features) ---")
    print(res)
    
    # 6. Visualization
    plt.figure(figsize=(12,6))
    # Using 'hue' to avoid warnings, setting legend=False
    sns.barplot(data=res, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    
    plt.title("Importance of SLSA v0.1 Provenance Fields")
    plt.xlabel("Importance Score (Gini Impurity)")
    plt.ylabel("Provenance Field")
    plt.tight_layout()
    
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig('results/feature_importance.png')
    print("\n[SUCCESS] Plot saved to 'results/feature_importance.png'")

if __name__ == "__main__":
    analyze()