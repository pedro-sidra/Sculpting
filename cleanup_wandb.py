import wandb

def cleanup_wandb_artifacts(entity: str, project: str, dry_run: bool = True):
    """
    Iterates through all runs in a project, keeps only the most recent 
    'model_last' artifact per run, and deletes all other model artifacts.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    print(f"Found {len(runs)} runs in {entity}/{project}.")
    print(f"Dry Run Mode: {'ENABLED (No deletions will occur)' if dry_run else 'DISABLED (Artifacts WILL be deleted)'}\n")

    for run in runs:
        print(f"--- Processing Run: {run.name} ({run.id}) ---")
        
        # 1. Fetch all model artifacts associated with the run
        model_artifacts = [art for art in run.logged_artifacts() if art.type == "model"]
        
        if not model_artifacts:
            print("  No model artifacts found. Skipping.")
            continue
            
        # 2. Isolate "model_last" artifacts to find the one to keep
        model_last_artifacts = [art for art in model_artifacts if "model_last" in art.name]
        
        latest_model_last = None
        if model_last_artifacts:
            # Sort chronologically by creation time (newest first)
            model_last_artifacts = sorted(model_last_artifacts, key=lambda x: x.created_at, reverse=True)
            latest_model_last = model_last_artifacts[0]
            print(f"  [KEEP] Latest 'model_last': {latest_model_last.name} (Created: {latest_model_last.created_at})")
        else:
            # Safe-guard: If a run doesn't have a model_last, skip to avoid deleting everything
            print("  [SKIP] No 'model_last' artifact found in this run. Skipping deletion to be safe.")
            continue

        # 3. Delete all OTHER model artifacts
        for artifact in model_artifacts:
            # Skip the single artifact we decided to keep
            if artifact.id == latest_model_last.id:
                continue 
                
            print(f"  [DELETE] Removing artifact: {artifact.name}")
            
            if not dry_run:
                try:
                    # 1. Clear any aliases (like 'latest') to unlock the artifact
                    if hasattr(artifact, 'aliases') and artifact.aliases:
                        artifact.aliases = []
                        artifact.save()
                    
                    # 2. Proceed with deletion
                    artifact.delete()
                    print(f"    -> Successfully deleted {artifact.name}")
                except Exception as e:
                    print(f"    -> Failed to delete {artifact.name}: {e}")

    print("\nCleanup complete.")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    ENTITY = "pedrosidra"
    PROJECT = "diss"
    
    # Keep dry_run=True for your first pass to verify the logic selects the correct artifacts. 
    # Change to False when you are ready to permanently free up disk space.
    cleanup_wandb_artifacts(entity=ENTITY, project=PROJECT, dry_run=False)