import os
import json
import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage

logger = logging.getLogger(__name__)

class CohortsStage(PipelineStage):
    """
    Identifies specific user cohorts for evaluation:
    - warm_start_users (e.g., > 10 interactions)
    - cold_start_users (e.g., <= 10 interactions)
    - fairness_evaluation groups
    """
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        
    @property
    def name(self) -> str:
        return "cohorts"
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute cohort assignments."""
        
        cleaned_filename = f"privacy_cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        filepath = os.path.join(self.artifacts_path, cleaned_filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cleaned dataset {filepath} not found for cohorts pass.")
            
        df = pd.read_parquet(filepath)
        
        logger.info(f"Generating cohorts for {dataset_name}...")
        
        # We need a user identifier column
        user_col = None
        for col in ['user_id', 'User_ID', 'id']:
            if col in df.columns:
                user_col = col
                break
                
        cohorts = {
            "warm_start_users": [],
            "cold_start_users": [],
            "fairness_groups": {}
        }
        
        if user_col:
            # Calculate interaction counts
            interaction_counts = df[user_col].value_counts()
            
            warm_users = interaction_counts[interaction_counts > 10].index.tolist()
            cold_users = interaction_counts[interaction_counts <= 10].index.tolist()
            
            # Since user_ids might be hashes or integers, we just store them
            # For massive datasets, we might only store a sample or aggregate statistics
            # Let's store aggregate stats to save space
            cohorts["warm_start_count"] = len(warm_users)
            cohorts["cold_start_count"] = len(cold_users)
            
            # Simulated fairness groups based on inferred categorical variables (if present)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if col != user_col and col != 'target_id':
                    # Only calculate fairness distribution for columns with small cardinality
                    if df[col].nunique() < 20:
                        cohorts["fairness_groups"][col] = df[col].value_counts().to_dict()
                        
        else:
            logger.warning(f"No valid user_id column found in {dataset_name} for cohort generation.")
            
        cohorts_filepath = os.path.join(self.artifacts_path, f"cohorts_{dataset_name}.json")
        with open(cohorts_filepath, "w") as f:
            json.dump(cohorts, f, indent=4, default=str)
            
        logger.info(f"Saved cohort profiles to {cohorts_filepath}")
        
        return {
            "cohorts_file": cohorts_filepath,
            "warm_start_count": cohorts.get("warm_start_count", 0),
            "cold_start_count": cohorts.get("cold_start_count", 0)
        }
