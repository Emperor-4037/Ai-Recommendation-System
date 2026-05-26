from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class DatasetManifest(Base):
    __tablename__ = 'dataset_manifests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    source_type = Column(String, nullable=False) # direct_interaction, proxy_only, synthetic
    raw_file_hashes = Column(JSON, nullable=False) # e.g. {'filename': 'hash'}
    row_count = Column(Integer)
    column_count = Column(Integer)
    dataset_routing_decision = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class SchemaFingerprint(Base):
    __tablename__ = 'schema_fingerprints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    manifest_id = Column(Integer, ForeignKey('dataset_manifests.id'), nullable=False)
    detected_columns = Column(JSON, nullable=False)
    inferred_semantic_roles = Column(JSON)
    timestamp_availability = Column(String) # present, absent, uncertain
    label_availability = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class PipelineCheckpoint(Base):
    __tablename__ = 'pipeline_checkpoints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stage_name = Column(String, nullable=False)
    dataset_name = Column(String, nullable=False)
    status = Column(String, nullable=False) # started, completed, failed
    artifacts_produced = Column(JSON) # e.g. {'cleaned_data': 'path/to/file'}
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

class ProvenanceRecord(Base):
    __tablename__ = 'provenance_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String, nullable=False) # row, column, dataset
    entity_id = Column(String, nullable=False) 
    source_fields = Column(JSON, nullable=False)
    transformation_applied = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DecisionLog(Base):
    __tablename__ = 'decision_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    context = Column(String, nullable=False)
    decision = Column(String, nullable=False)
    consequences = Column(String)
    status = Column(String, default="proposed")
    created_at = Column(DateTime, default=datetime.utcnow)
