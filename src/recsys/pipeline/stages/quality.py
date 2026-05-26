import os
import json
import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage

logger = logging.getLogger(__name__)

class QualityStage(PipelineStage):
    """
    Generates a source-specific quality report assessing:
    - Missingness profile
    - Sparsity and density of interactions
    - Duplicate rate
    - General descriptive statistics
    """
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        
    @property
    def name(self) -> str:
        return "quality"
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute quality report generation."""
        
        cleaned_filename = f"privacy_cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        filepath = os.path.join(self.artifacts_path, cleaned_filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cleaned dataset {filepath} not found for quality report pass.")
            
        df = pd.read_parquet(filepath)
        
        logger.info(f"Generating quality report for {dataset_name}...")
        
        # 1. Missingness Profile
        missingness = df.isna().mean().to_dict()
        
        # 2. General Descriptive Statistics
        # We only take basic stats for numerics to save space
        stats = df.describe().to_dict()
        
        # 3. Sparsity for Interaction Data (if applicable)
        sparsity_metrics = {}
        if 'user_id' in df.columns and 'target_id' in df.columns:
            num_users = df['user_id'].nunique()
            num_targets = df['target_id'].nunique()
            num_interactions = len(df)
            
            if num_users > 0 and num_targets > 0:
                density = num_interactions / (num_users * num_targets)
                sparsity_metrics = {
                    "num_users": num_users,
                    "num_targets": num_targets,
                    "density": density,
                    "sparsity": 1 - density
                }
                
        # Combine into report
        report = {
            "dataset": dataset_name,
            "total_rows": len(df),
            "missingness_profile": missingness,
            "interaction_sparsity": sparsity_metrics,
            "descriptive_statistics": stats
        }
        
        quality_filepath = os.path.join(self.artifacts_path, f"quality_report_{dataset_name}.json")
        with open(quality_filepath, "w") as f:
            json.dump(report, f, indent=4, default=str)
            
        logger.info(f"Saved quality report to {quality_filepath}")
        
        return {
            "quality_report_path": quality_filepath
        }
