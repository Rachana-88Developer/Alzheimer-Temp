"""
Alzheimer's Disease Dataset Loader
Downloads directly from Kaggle URL: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
"""
import os
# Force Kaggle to download to D drive project folder
os.environ["KAGGLEHUB_CACHE"] = r"d:\IEEE_ML\cache"

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KAGGLE_URL = "rabieelkharoua/alzheimers-disease-dataset"


def load_and_preprocess():
    """Download dataset from Kaggle URL and return preprocessed train/test splits."""

    # --- Step 1: Download from Kaggle URL ---
    print("=" * 55)
    print("  DOWNLOADING ALZHEIMER'S DATASET FROM KAGGLE")
    print("=" * 55)
    path = kagglehub.dataset_download(KAGGLE_URL)
    print(f"[OK] Downloaded to: {path}\n")

    # --- Step 2: Load CSV (auto-detect filename) ---
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    csv_file = os.path.join(path, csv_files[0])
    print(f"[FILE] Found: {csv_files[0]}")
    df = pd.read_csv(csv_file)

    print(f"[DATA] Dataset Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[INFO] Target Classes    : {df['Diagnosis'].value_counts().to_dict()}")
    print(f"       0 = No Alzheimer's, 1 = Alzheimer's\n")

    # --- Step 3: Drop non-informative columns ---
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

    # --- Step 4: Separate features & target ---
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    feature_names = X.columns.tolist()

    # --- Step 5: Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Step 6: Train-Test split (80/20, stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[TRAIN] Training samples  : {X_train.shape[0]}")
    print(f"[TEST]  Testing samples   : {X_test.shape[0]}")
    print(f"[FEAT]  Total features    : {X_train.shape[1]}")
    print(f"[FEAT]  Feature names     : {feature_names[:5]}... (+{len(feature_names)-5} more)")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess()
    print("\n[OK] Dataset loaded and preprocessed successfully!")
