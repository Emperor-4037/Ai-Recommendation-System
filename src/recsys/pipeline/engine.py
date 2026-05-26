import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from recsys.db.models import PipelineCheckpoint

logger = logging.getLogger(__name__)

class PipelineStage(ABC):
    """Base class for an idempotent pipeline stage."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the stage."""
        pass
        
    @abstractmethod
    async def execute(self, session: AsyncSession, dataset_name: str, force_rebuild: bool) -> Dict[str, Any]:
        """Execute the stage and return artifacts produced."""
        pass

class PipelineEngine:
    """Lightweight DAG executor adhering to manifest and checkpoint constraints."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.stages: List[PipelineStage] = []
        
    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)
        
    async def get_latest_checkpoint(self, stage_name: str, dataset_name: str) -> Optional[PipelineCheckpoint]:
        stmt = select(PipelineCheckpoint).where(
            PipelineCheckpoint.stage_name == stage_name,
            PipelineCheckpoint.dataset_name == dataset_name,
            PipelineCheckpoint.status == "completed"
        ).order_by(PipelineCheckpoint.completed_at.desc()).limit(1)
        
        result = await self.session.execute(stmt)
        return result.scalars().first()
        
    async def record_checkpoint(self, stage_name: str, dataset_name: str, status: str, artifacts: Optional[Dict[str, Any]] = None):
        checkpoint = PipelineCheckpoint(
            stage_name=stage_name,
            dataset_name=dataset_name,
            status=status,
            artifacts_produced=artifacts,
            started_at=datetime.utcnow() if status == "started" else None,
            completed_at=datetime.utcnow() if status == "completed" else None
        )
        self.session.add(checkpoint)
        await self.session.commit()

    async def run(self, dataset_name: str, resume: bool = False, force_rebuild: bool = False):
        """Execute all stages sequentially."""
        logger.info(f"Starting pipeline for dataset: {dataset_name}. Resume: {resume}, Force: {force_rebuild}")
        
        for stage in self.stages:
            logger.info(f"Checking stage: {stage.name}")
            
            if not force_rebuild:
                checkpoint = await self.get_latest_checkpoint(stage.name, dataset_name)
                if checkpoint and resume:
                    logger.info(f"Stage {stage.name} already completed at {checkpoint.completed_at}. Skipping.")
                    continue
            
            logger.info(f"Executing stage: {stage.name}")
            await self.record_checkpoint(stage.name, dataset_name, "started")
            
            try:
                artifacts = await stage.execute(self.session, dataset_name, force_rebuild)
                await self.record_checkpoint(stage.name, dataset_name, "completed", artifacts)
                logger.info(f"Stage {stage.name} completed successfully.")
            except Exception as e:
                await self.record_checkpoint(stage.name, dataset_name, "failed")
                logger.error(f"Stage {stage.name} failed: {str(e)}")
                raise e
