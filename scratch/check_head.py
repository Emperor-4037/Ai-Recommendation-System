import pandas as pd
df = pd.read_parquet(r"d:\Ai Recommendation System\data\artifacts\cleaned_rec-libimseti-dir.parquet")
print(df.head(20))
