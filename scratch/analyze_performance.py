import optuna

def analyze_performance():
    try:
        study = optuna.load_study(study_name="study_rec-libimseti-dir.edges", storage="sqlite:///hyperopt_study.db")
        print("Tuning Objective: Maximize Offline Ranking NDCG@10\n")
        
        # Check if completed
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        print(f"Study Status Summary:")
        print(f"  Total trials in DB: {len(study.trials)}")
        print(f"  Completed trials: {len(completed_trials)}")
        print(f"  Pruned trials: {len(pruned_trials)}")
        print(f"  Running trials: {len(running_trials)}")
        print(f"  Failed trials: {len(failed_trials)}")
        if study.best_trial:
            print(f"  Best Trial Number: {study.best_trial.number}")
            print(f"  Best NDCG@10 (Value): {study.best_trial.value:.6f}")
            print(f"  Best Hyperparameters: {study.best_trial.params}")
        print("\nAll Trials details:")
        
        for t in study.trials:
            dur_str = f"{(t.datetime_complete - t.datetime_start).total_seconds():.1f}s" if (t.datetime_start and t.datetime_complete) else "N/A"
            print(f"Trial {t.number} ({t.state.name}): Duration={dur_str}, Value={t.value if t.value is not None else 'N/A'}")
            print(f"  Params: {t.params}")
            print(f"  Intermediate Values (Epoch -> -Val Loss): {dict(t.intermediate_values)}")
            
    except Exception as e:
        print("Error reading database:", e)

if __name__ == "__main__":
    analyze_performance()
