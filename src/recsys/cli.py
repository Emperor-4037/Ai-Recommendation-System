import click
import asyncio

@click.group()
def cli():
    """AI Recommendation System CLI"""
    pass

@cli.command()
@click.argument('dataset_names', nargs=-1)
@click.option('--resume', is_flag=True, help="Continue from latest valid checkpoint")
@click.option('--force-rebuild', is_flag=True, help="Recompute all derived artifacts from raw inputs")
def run(dataset_names, resume, force_rebuild):
    """Run the pipeline on specific datasets."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not dataset_names:
        dataset_names = ["okcupid_profiles.csv", "rec-libimseti-dir.edges", "dating_app_behavior_dataset_extended1.csv"]
    
    click.echo(f"Running pipeline for {dataset_names}. Resume: {resume}, Force: {force_rebuild}")
    
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    async def _run():
        from recsys.db.core import AsyncSessionLocal
        from recsys.pipeline.engine import PipelineEngine
        from recsys.pipeline.stages.ingestion import DataIngestionStage
        from recsys.pipeline.stages.preprocessing import PreprocessingStage
        from recsys.pipeline.stages.privacy import PrivacyStage
        from recsys.pipeline.stages.splitting import SplittingStage
        from recsys.pipeline.stages.cohorts import CohortsStage
        from recsys.pipeline.stages.feature_engineering import FeatureEngineeringStage
        from recsys.pipeline.stages.provenance import ProvenanceStage
        from recsys.pipeline.stages.quality import QualityStage
        from recsys.pipeline.stages.reporting import ReportingStage
        import os
        
        # Hardcoding the paths for now based on implementation plan
        RAW_DATA_PATH = r"d:\Ai Recommendation System\datasets"
        ARTIFACTS_PATH = r"d:\Ai Recommendation System\data\artifacts"
        
        async with AsyncSessionLocal() as session:
            engine = PipelineEngine(session)
            engine.add_stage(DataIngestionStage(RAW_DATA_PATH))
            engine.add_stage(PreprocessingStage(RAW_DATA_PATH, ARTIFACTS_PATH))
            engine.add_stage(PrivacyStage(ARTIFACTS_PATH))
            engine.add_stage(SplittingStage(ARTIFACTS_PATH))
            engine.add_stage(CohortsStage(ARTIFACTS_PATH))
            engine.add_stage(FeatureEngineeringStage(ARTIFACTS_PATH))
            engine.add_stage(ProvenanceStage(ARTIFACTS_PATH))
            engine.add_stage(QualityStage(ARTIFACTS_PATH))
            engine.add_stage(ReportingStage(ARTIFACTS_PATH))
            
            for dataset in dataset_names:
                try:
                    await engine.run(dataset, resume, force_rebuild)
                    click.echo(f"Pipeline completed successfully for {dataset}")
                except Exception as e:
                    click.echo(f"Pipeline failed for {dataset}: {e}")
                    
    asyncio.run(_run())
    
@cli.command()
@click.option('--dataset', required=True, help="Dataset name to train on")
@click.option('--batch-size', type=int, default=None, help="Override training batch size")
@click.option('--epochs', type=int, default=None, help="Override training epochs")
@click.option('--lr', type=float, default=None, help="Override training learning rate")
def train(dataset, batch_size, epochs, lr):
    """Train the champion models and export them."""
    import sys
    import asyncio
    import logging
    import optuna
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    from recsys.operations.train import train_pipeline
    from recsys.operations.export import export_artifacts
    import click
    
    click.echo(f"Starting final training on {dataset}")
    
    best_params = None
    try:
        study = optuna.load_study(study_name=f"study_{dataset}", storage="sqlite:///hyperopt_study.db")
        best_params = study.best_params
        click.echo(f"Found champion parameters from Optuna study: {best_params}")
    except Exception as e:
        click.echo(f"Could not load Optuna study, falling back to defaults. ({e})")
        
    if best_params is None:
        best_params = {}
        
    # Apply CLI overrides if provided
    if batch_size is not None:
        best_params['batch_size'] = batch_size
        click.echo(f"Overriding batch size to {batch_size}")
    if epochs is not None:
        best_params['epochs'] = epochs
        click.echo(f"Overriding epochs to {epochs}")
    if lr is not None:
        best_params['learning_rate'] = lr
        click.echo(f"Overriding learning rate to {lr}")
        
    retrieval_model, scorer_model, reranker_model, calibrator, _, faiss_index, model_config = train_pipeline(dataset, params=best_params)
    export_artifacts(retrieval_model, scorer_model, reranker_model, calibrator, {"status": "champion"}, params=best_params, faiss_index=faiss_index, model_config=model_config)
    click.echo("Training and export complete.")

@cli.command()
@click.option('--dataset', required=True, help="Dataset name to tune on")
@click.option('--n-trials', default=2, help="Number of optuna trials")
def tune(dataset, n_trials):
    """Run hyperparameter search via Optuna."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from recsys.operations.hyperopt import HyperparameterOptimizer, get_objective
    import click
    
    click.echo(f"Starting tuning on {dataset} with {n_trials} trials")
    optimizer = HyperparameterOptimizer(study_name=f"study_{dataset}", n_trials=n_trials)
    best_params = optimizer.optimize(get_objective(dataset))
    click.echo(f"Tuning complete. Best params: {best_params}")

@cli.command()
@click.option('--dataset', required=True, help="Dataset name to evaluate")
@click.option('--max-users', default=5000, help="Max users to evaluate")
@click.option('--k', default=10, help="Top-k cutoff for metrics")
@click.option('--retrieval-k', default=200, help="Number of candidates to retrieve per user")
def evaluate(dataset, max_users, k, retrieval_k):
    """Evaluate the trained model end-to-end in Sync and Spark modes."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from recsys.operations.evaluate import evaluate_end_to_end

    click.echo(f"Starting end-to-end evaluation on {dataset}")
    results = evaluate_end_to_end(dataset, max_users=max_users, k=k, retrieval_k=retrieval_k)
    click.echo("Evaluation complete.")

if __name__ == '__main__':
    cli()

