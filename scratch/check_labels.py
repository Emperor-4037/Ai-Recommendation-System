import pandas as pd
import numpy as np

filepath = r"d:\Ai Recommendation System\data\artifacts\splits_privacy_cleaned_rec-libimseti-dir.parquet"
df = pd.read_parquet(filepath)
print("Columns:", df.columns)
print("DF Length:", len(df))
print("Split value counts:")
print(df['split'].value_counts() if 'split' in df.columns else "No split column")

label_cols = ['match', 'rating', 'label', 'swiped_right']
for col in label_cols:
    if col in df.columns:
        s = df[col]
        print(f"\nStats for column: {col}")
        print(f"dtype: {s.dtype}")
        print(f"min: {s.min()}, max: {s.max()}")
        print(f"NaN count: {s.isna().sum()}")
        print(f"Unique values: {s.unique()[:20]}")
