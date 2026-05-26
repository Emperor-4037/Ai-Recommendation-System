import pandas as pd
import os

folder = r"d:\Ai Recommendation System\data\artifacts"
files = [
    "cleaned_rec-libimseti-dir.parquet",
    "privacy_cleaned_rec-libimseti-dir.parquet",
    "splits_privacy_cleaned_rec-libimseti-dir.parquet"
]

for f in files:
    path = os.path.join(folder, f)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"\n--- File: {f} ---")
        print("Columns:", list(df.columns))
        print("Rows:", len(df))
        if 'rating' in df.columns:
            s = df['rating']
            print("Rating Stats:")
            print("  min:", s.min())
            print("  max:", s.max())
            print("  isna count:", s.isna().sum())
            print("  unique values:", s.unique()[:10])
        else:
            print("rating column NOT found")
    else:
        print(f"\nFile {f} does not exist")
