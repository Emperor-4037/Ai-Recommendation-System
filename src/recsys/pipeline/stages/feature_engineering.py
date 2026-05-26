import os
import json
import joblib
import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.preprocessing import StandardScaler

from recsys.pipeline.engine import PipelineStage

logger = logging.getLogger(__name__)

class FeatureEngineeringStage(PipelineStage):
    """
    Builds feature vocabularies and fits scalers strictly on the training split.
    """
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        
    @property
    def name(self) -> str:
        return "feature_engineering"
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute feature engineering logic."""
        
        cleaned_filename = f"privacy_cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        split_filepath = os.path.join(self.artifacts_path, f"splits_{cleaned_filename}")
        
        if not os.path.exists(split_filepath):
            raise FileNotFoundError(f"Splits dataset {split_filepath} not found.")
            
        df = pd.read_parquet(split_filepath)
        
        logger.info(f"Extracting vocabularies and fitting scalers for {dataset_name}...")
        
        # Strictly use the training split to avoid leakage
        train_df = df[df['split'] == 'train']
        
        if train_df.empty:
            logger.warning("Training split is empty. Cannot fit scalers/vocabularies.")
            return {}
            
        vocabularies = {}
        scalers = {}
        
        # Define simplistic feature types
        categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = train_df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove metadata columns from feature processing
        ignored = ['user_id', 'target_id', 'User_ID', 'split']
        categorical_cols = [c for c in categorical_cols if c not in ignored]
        numerical_cols = [c for c in numerical_cols if c not in ignored]
        
        # 1. Feature Vocabularies
        for col in categorical_cols:
            unique_vals = train_df[col].dropna().unique().tolist()
            # Sort for determinism
            unique_vals.sort()
            vocabularies[col] = unique_vals
            
        vocab_path = os.path.join(self.artifacts_path, f"vocab_{dataset_name}.json")
        with open(vocab_path, "w") as f:
            json.dump(vocabularies, f, indent=4)
            
        # 2. Fitted Scalers
        for col in numerical_cols:
            scaler = StandardScaler()
            # Handle potential NaNs just in case, though preprocessing should have handled them
            vals = train_df[[col]].dropna()
            if not vals.empty:
                scaler.fit(vals)
                scalers[col] = scaler
                
        scaler_path = os.path.join(self.artifacts_path, f"scalers_{dataset_name}.joblib")
        joblib.dump(scalers, scaler_path)
        
        logger.info(f"Saved feature vocabularies to {vocab_path}")
        logger.info(f"Saved fitted scalers to {scaler_path}")
        
        return {
            "vocabularies_path": vocab_path,
            "scalers_path": scaler_path,
            "categorical_features": categorical_cols,
            "numerical_features": numerical_cols
        }
