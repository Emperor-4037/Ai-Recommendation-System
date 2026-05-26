import os
import hashlib
import pandas as pd
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage
from recsys.db.models import DatasetManifest, SchemaFingerprint

def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class DataIngestionStage(PipelineStage):
    """
    Ingests raw data files, computes hashes, creates manifests, 
    and applies source routing rules.
    """
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        
    @property
    def name(self) -> str:
        return "ingestion"
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute ingestion and schema detection."""
        
        # 1. Locate file
        # We will assume dataset_name matches the file name in the raw_data_path
        filepath = os.path.join(self.raw_data_path, dataset_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file {filepath} not found.")
            
        # 2. Compute Hashes
        file_hash = calculate_file_hash(filepath)
        raw_file_hashes = {dataset_name: file_hash}
        
        # 3. Source Routing Logic
        source_type = "ambiguous"
        routing_decision = "default_routing"
        
        if "libimseti" in dataset_name.lower():
            source_type = "directed_user_user_rating_graph"
            routing_decision = "preserve rating signal; pairwise only"
        elif "okcupid" in dataset_name.lower():
            source_type = "profile_only_proxy_pretraining"
            routing_decision = "forbid fabricated labels; proxy only"
        elif "kaggle_dating_app" in dataset_name.lower() or "behavior_dataset" in dataset_name.lower():
            source_type = "synthetic_supervised_stress_test"
            routing_decision = "isolate from real data; use for stress tests"
        
        # 4. Schema inference via chunked reading (hardware budget aware)
        columns = []
        row_count = 0
        if filepath.endswith('.csv'):
            chunk_iter = pd.read_csv(filepath, chunksize=10000)
            first_chunk = next(chunk_iter)
            columns = list(first_chunk.columns)
            row_count += len(first_chunk)
            for chunk in chunk_iter:
                row_count += len(chunk)
        elif filepath.endswith('.edges'):
            # Specific to Libimseti
            chunk_iter = pd.read_csv(filepath, sep='\t', comment='%', names=['user_id', 'target_id', 'rating'], chunksize=10000)
            first_chunk = next(chunk_iter)
            columns = list(first_chunk.columns)
            row_count += len(first_chunk)
            for chunk in chunk_iter:
                row_count += len(chunk)
                
        detected_columns = {col: "inferred_string_or_numeric" for col in columns}
        
        # 5. Emit Manifest
        manifest = DatasetManifest(
            dataset_name=dataset_name,
            version="1.0",
            source_type=source_type,
            raw_file_hashes=raw_file_hashes,
            row_count=row_count,
            column_count=len(columns),
            dataset_routing_decision=routing_decision
        )
        session.add(manifest)
        await session.flush() # flush to get manifest.id
        
        # 6. Emit Schema Fingerprint
        schema_fingerprint = SchemaFingerprint(
            manifest_id=manifest.id,
            detected_columns=detected_columns,
            inferred_semantic_roles={"features": columns},
            timestamp_availability="uncertain", # Default rule from prompt
            label_availability="uncertain"
        )
        session.add(schema_fingerprint)
        await session.commit()
        
        return {
            "manifest_id": manifest.id,
            "schema_fingerprint_id": schema_fingerprint.id,
            "row_count": row_count,
            "source_type": source_type
        }
