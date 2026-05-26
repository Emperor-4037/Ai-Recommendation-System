import os
import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage

logger = logging.getLogger(__name__)

class PreprocessingStage(PipelineStage):
    """
    Deterministically cleans data:
    - deduplication
    - standardize categories
    - cap outliers
    - impute conservatively
    """
    
    def __init__(self, raw_data_path: str, artifacts_path: str):
        self.raw_data_path = raw_data_path
        self.artifacts_path = artifacts_path
        os.makedirs(self.artifacts_path, exist_ok=True)
        
    @property
    def name(self) -> str:
        return "preprocessing"
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute deterministic cleaning rules."""
        
        filepath = os.path.join(self.raw_data_path, dataset_name)
        cleaned_filename = f"cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        cleaned_filepath = os.path.join(self.artifacts_path, cleaned_filename)
        
        logger.info(f"Preprocessing {dataset_name} in CPU chunks...")
        
        # Determine source type to apply source-specific cleaning
        if "libimseti" in dataset_name.lower():
            # Edge list
            chunk_iter = pd.read_csv(filepath, sep='\t', comment='%', names=['user_id', 'target_id', 'rating'], chunksize=100000)
            cleaned_chunks = []
            for chunk in chunk_iter:
                # Deterministic dedup: keep first observed edge
                chunk = chunk.drop_duplicates(subset=['user_id', 'target_id'], keep='first')
                # Cap ratings to a logical max if needed
                # (Libimseti ratings are 1-10)
                chunk['rating'] = chunk['rating'].clip(lower=1, upper=10)
                cleaned_chunks.append(chunk)
            df = pd.concat(cleaned_chunks)
            
        elif "okcupid" in dataset_name.lower() or "kaggle_dating_app" in dataset_name.lower() or "behavior_dataset" in dataset_name.lower():
            # We assume CSV with headers
            chunk_iter = pd.read_csv(filepath, chunksize=50000)
            cleaned_chunks = []
            for chunk in chunk_iter:
                # Deterministic dedup
                chunk = chunk.drop_duplicates()
                
                # Standardize categories: strip and lower strings
                for col in chunk.select_dtypes(include=['object']).columns:
                    chunk[col] = chunk[col].astype(str).str.strip().str.lower()
                    
                # Fill NAs conservatively
                for col in chunk.select_dtypes(include=['number']).columns:
                    chunk[col] = chunk[col].fillna(chunk[col].median() if not chunk[col].isna().all() else 0)
                    
                cleaned_chunks.append(chunk)
            df = pd.concat(cleaned_chunks)
        else:
            raise ValueError(f"Unknown source routing for {dataset_name}")
            
        # Hardware policy: write as serialized/compact format
        df.to_parquet(cleaned_filepath, index=False)
        logger.info(f"Saved cleaned dataset to {cleaned_filepath}")
        
        return {
            "cleaned_data_path": cleaned_filepath,
            "final_row_count": len(df)
        }
