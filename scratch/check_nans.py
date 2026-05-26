import pandas as pd
import numpy as np

path = "d:/Ai Recommendation System/data/artifacts/splits_privacy_cleaned_rec-libimseti-dir.parquet"
df = pd.read_parquet(path)

print(f"Columns: {df.columns.tolist()}")

for col in ['rating', 'label', 'u_likes_c', 'c_likes_u', 'match', 'propensity']:
    if col in df.columns:
        nans = df[col].isna().sum()
        print(f"'{col}': {nans} NaNs. Min: {df[col].min()}, Max: {df[col].max()}")
