import os
import hashlib
import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage

logger = logging.getLogger(__name__)

class PrivacyStage(PipelineStage):
    """
    Implements privacy and provenance rules:
    - stable_salted_hash_user_ids
    - separates PII from modeling tables
    """
    
    def __init__(self, artifacts_path: str, salt: str = "default_secure_salt_2026"):
        self.artifacts_path = artifacts_path
        self.salt = salt
        
    @property
    def name(self) -> str:
        return "privacy"
        
    def _hash_id(self, val: Any) -> str:
        """Stable salted hash for user ids."""
        salted_str = f"{self.salt}_{val}"
        return hashlib.sha256(salted_str.encode('utf-8')).hexdigest()[:16]
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Apply privacy-safe hashing and drop direct identifiers."""
        
        cleaned_filename = f"cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        filepath = os.path.join(self.artifacts_path, cleaned_filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cleaned dataset {filepath} not found for privacy pass.")
            
        df = pd.read_parquet(filepath)
        
        logger.info(f"Applying privacy constraints for {dataset_name}...")
        
        # 1. Identify user/target ID columns
        id_cols = [c for c in df.columns if c.lower() in ['user_id', 'target_id', 'id']]
        
        for col in id_cols:
            # Optimization: Hash only unique values and map them back (O(unique_users) instead of O(rows))
            unique_ids = df[col].unique()
            hashed_unique = {val: self._hash_id(val) for val in unique_ids}
            df[col] = df[col].map(hashed_unique)
            
        # 2. PII Separation
        # Keep an explicit allowlist of non-PII categorical and numerical features
        # If there are text columns or raw names, we drop them
        pii_keywords = ['name', 'email', 'phone', 'address', 'ip', 'username', 'location']
        dropped_cols = []
        
        for col in df.columns:
            if any(pii in col.lower() for pii in pii_keywords):
                df = df.drop(columns=[col])
                dropped_cols.append(col)
                
        # 3. Save privacy-safe canonical version
        privacy_filepath = os.path.join(self.artifacts_path, f"privacy_{cleaned_filename}")
        df.to_parquet(privacy_filepath, index=False)
        
        logger.info(f"Saved privacy-safe dataset to {privacy_filepath}")
        
        return {
            "privacy_data_path": privacy_filepath,
            "hashed_columns": id_cols,
            "dropped_pii_columns": dropped_cols
        }
