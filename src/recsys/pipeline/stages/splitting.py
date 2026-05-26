import os
import logging
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage
from recsys.db.models import DatasetManifest

logger = logging.getLogger(__name__)

class SplittingStage(PipelineStage):
    """
    Deterministically splits data while preventing leakage.
    - Time-based if timestamps are trustworthy
    - Stable-hash fallback
    """
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        
    @property
    def name(self) -> str:
        return "splitting"
        
    def _stable_hash_split(self, df: pd.DataFrame, group_col: str, train_pct=0.8, val_pct=0.1) -> pd.DataFrame:
        """Assigns splits based on a stable hash of the group_col to prevent leakage."""
        def get_split(val):
            # MD5 hash is stable across runs and platforms
            h = int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16)
            # Modulo 100 to get a percentage
            pct = h % 100
            if pct < train_pct * 100:
                return 'train'
            elif pct < (train_pct + val_pct) * 100:
                return 'val'
            else:
                return 'test'
                
        return df[group_col].apply(get_split)
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute splitting logic."""
        
        cleaned_filename = f"privacy_cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        cleaned_filepath = os.path.join(self.artifacts_path, cleaned_filename)
        
        if not os.path.exists(cleaned_filepath):
            raise FileNotFoundError(f"Cleaned dataset {cleaned_filepath} not found.")
            
        df = pd.read_parquet(cleaned_filepath)
        
        logger.info(f"Splitting {dataset_name}...")
        
        # Source-specific splitting logic
        if "libimseti" in dataset_name.lower():
            # Pair-level leakage prevention: hash the sorted pair tuple
            pair_df = df[['user_id', 'target_id']].astype(str)
            min_vals = pair_df.min(axis=1)
            max_vals = pair_df.max(axis=1)
            
            df['pair_id'] = min_vals + "_" + max_vals
            df['split'] = self._stable_hash_split(df, 'pair_id')
            df = df.drop(columns=['pair_id'])
            
        elif "okcupid" in dataset_name.lower():
            # User-level proxy target splitting (prevent user leakage)
            # Assuming okcupid dataset has an implicit user index if no user_id is found
            if 'user_id' not in df.columns:
                df['user_id'] = np.arange(len(df))
            df['split'] = self._stable_hash_split(df, 'user_id')
            
        elif "kaggle_dating_app" in dataset_name.lower() or "behavior_dataset" in dataset_name.lower():
            # Assuming synthetic dataset has User_ID column
            group_col = 'User_ID' if 'User_ID' in df.columns else 'user_id'
            if group_col not in df.columns:
                df[group_col] = np.arange(len(df))
            df['split'] = self._stable_hash_split(df, group_col)
        else:
            raise ValueError(f"Unknown source routing for splitting {dataset_name}")
            
        # Save splits
        split_filepath = os.path.join(self.artifacts_path, f"splits_{cleaned_filename}")
        df.to_parquet(split_filepath, index=False)
        logger.info(f"Saved splits to {split_filepath}")
        
        split_counts = df['split'].value_counts().to_dict()
        
        return {
            "split_data_path": split_filepath,
            "split_counts": split_counts
        }
