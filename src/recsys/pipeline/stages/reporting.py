import os
import json
import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.pipeline.engine import PipelineStage

logger = logging.getLogger(__name__)

class ReportingStage(PipelineStage):
    """
    Generates final human-readable reports and machine-readable metadata.
    """
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        
    @property
    def name(self) -> str:
        return "reporting"
        
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Generate reports."""
        
        # We will assume splitting was done.
        cleaned_filename = f"privacy_cleaned_{dataset_name}"
        if not cleaned_filename.endswith('.parquet'):
            cleaned_filename = cleaned_filename.rsplit('.', 1)[0] + '.parquet'
            
        report_data = {
            "dataset_name": dataset_name,
            "status": "Ready",
            "source_type_inferred": True,
            "pipeline_stages_run": ["ingestion", "preprocessing", "splitting"],
            "artifact_locations": {
                "cleaned_data": os.path.join(self.artifacts_path, cleaned_filename),
                "splits_data": os.path.join(self.artifacts_path, f"splits_{cleaned_filename}")
            }
        }
        
        # Write machine-readable JSON
        json_report_path = os.path.join(self.artifacts_path, f"metadata_{dataset_name}.json")
        with open(json_report_path, "w") as f:
            json.dump(report_data, f, indent=4)
            
        # Write human-readable markdown
        md_report_path = os.path.join(self.artifacts_path, f"report_{dataset_name}.md")
        with open(md_report_path, "w") as f:
            f.write(f"# Preprocessing Report: {dataset_name}\n\n")
            f.write(f"**Status**: {report_data['status']}\n")
            f.write("## Artifacts\n")
            for k, v in report_data["artifact_locations"].items():
                f.write(f"- **{k}**: `{v}`\n")
                
        logger.info(f"Generated reports at {md_report_path} and {json_report_path}")
        
        return {
            "metadata_json": json_report_path,
            "report_md": md_report_path
        }
