import os
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    app_name: str = "Reciprocal Recommender"
    environment: str = "development"
    debug: bool = False
    
    # DB URLs
    postgres_url: str = Field(..., env="POSTGRES_URL")
    redis_url: str = Field(..., env="REDIS_URL")
    
    # Data & Artifacts
    artifact_path: str = Field(default="data/artifacts")
    dataset_path: str = Field(default="data/datasets")
    manifest_path: str = Field(default="data/manifests")
    model_path: str = Field(default="data/models")
    
    # Model Hyperparameters

    retrieval_embedding_dim: int = 64
    retrieval_hidden_sizes: list[int] = [256, 128, 64]
    scorer_hidden_sizes: list[int] = [256, 128, 64]
    scorer_dropout: float = 0.2
    
    # Spark Reranker & Policy thresholds
    lambda_spark: float = 0.2
    lambda_comp: float = 0.2
    lambda_risk: float = 1.0
    lambda_pop: float = 0.1
    min_reciprocal_score: float = 0.4
    max_safety_risk: float = 0.2
    
    # Phase 2 Advanced Configs
    sequence_hidden_dim: int = 64
    sequence_window_size: int = 10
    graph_embedding_dim: int = 32
    propensity_clipping: float = 0.05
    exploration_budget: float = 0.1
    multitask_weights: dict[str, float] = {
        "forward": 1.0, 
        "reverse": 1.0, 
        "mutual": 2.0, 
        "reply": 0.5, 
        "utility": 0.1
    }
    
    # MLOps & Hyperopt
    mlflow_tracking_uri: Optional[str] = None
    optuna_n_trials: int = 20
    canary_percentage: float = 0.1
    
    # Serving settings
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

_settings = None

def load_yaml_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        yaml_path = os.getenv("RECSYS_CONFIG_PATH", "config.yaml")
        yaml_settings = load_yaml_config(yaml_path)
        
        # Merge YAML into env settings if not explicitly passed as ENV overrides
        # Pydantic Settings reads ENV vars first, so we'll construct the object
        # with yaml defaults overridden by env.
        _settings = Settings(**yaml_settings)
    return _settings
