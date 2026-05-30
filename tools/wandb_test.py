import os
import yaml
import wandb
import pandas as pd
import tempfile
import subprocess
from contextlib import contextmanager

def save_run_config(api: wandb.Api, entity: str, project: str, run_id: str, base_output_dir: str = "./run_data") -> str:
    """
    Fetches and saves only the config for a specific WandB run.
    """
    print(f"--- Fetching config for Run ID: {run_id} ---")
    
    run_path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Could not find run {run_path}. Check entity, project, and run_id.") from e

    # Create an output directory specifically for this run's config
    run_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Dump the config to a local YAML file
    config_path = os.path.join(run_dir, f"config_{run_id}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(run.config, f, default_flow_style=False)
    print(f"[+] Saved config to: {config_path}")
    
    return config_path

@contextmanager
def download_temp_weights(api: wandb.Api, entity: str, project: str, run_id: str):
    """
    Context manager that downloads the latest model artifact to a temporary directory,
    temporarily overrides the WANDB_CACHE_DIR to prevent hidden disk bloat,
    yields the path to the .pth file, and automatically cleans up afterwards.
    """
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)
    
    artifacts = run.logged_artifacts()
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if not model_artifacts:
        raise ValueError(f"No artifacts of type 'model' found for run {run_id}.")
        
    artifact = max(model_artifacts, key=lambda a: a.created_at)
    
    # Create a temporary directory that automatically deletes itself when done
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override wandb cache dir so it doesn't cache large models in ~/.local/share/wandb
        original_cache = os.environ.get("WANDB_CACHE_DIR")
        os.environ["WANDB_CACHE_DIR"] = temp_dir
        
        try:
            print(f"[*] Downloading artifact '{artifact.name}' to temporary dir...")
            downloaded_path = artifact.download(root=temp_dir)
            
            weight_file_path = None
            for file in os.listdir(downloaded_path):
                if file.endswith(".pth"):
                    weight_file_path = os.path.join(downloaded_path, file)
                    break
                    
            if not weight_file_path:
                raise FileNotFoundError(f"No .pth file found in: {downloaded_path}")
            
            # Yield the weight file path so the test script can use it
            yield weight_file_path
            
        finally:
            # Restore the original cache environment variable
            if original_cache is not None:
                os.environ["WANDB_CACHE_DIR"] = original_cache
            else:
                del os.environ["WANDB_CACHE_DIR"]
            
            print(f"[*] Cleaning up temporary weights and cache...")

def process_and_build_experiments_df(entity: str, project: str, base_output_dir: str = "./run_data") -> pd.DataFrame:
    """
    Queries all runs, identifies relationships, tests the models using temporary 
    weight files, and builds a DataFrame summarizing the runs.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    data = []
    lr_tags = ["LR1", "LR5", "LR10", "LR20", "LR100"]
    
    for run in runs:
        # Check for pretrain tag case-insensitively just to be safe
        if any(tag.lower() == "pretrain" for tag in run.tags):
            continue
            
        # Detect LR split case-insensitively
        lr_split = next((tag.upper() for tag in run.tags if tag.upper() in lr_tags), None)
        
        parent_run_id = None
        parent_run_name = None
        try:
            for artifact in run.used_artifacts():
                creator = artifact.logged_by()
                if creator is not None and creator.project == project:
                    parent_run_id = creator.id
                    parent_run_name = creator.name
                    break
        except Exception as e:
            print(f"[!] Could not determine parent for run {run.id}: {e}")
            
        print(f"\n========================================")
        print(f"Processing Finetune Run: {run.id}")
        print(f"LR Split: {lr_split} | Parent Run: {parent_run_id}")
        print(f"========================================")
        
        # Extract summary metrics (last values logged) for all numerical fields
        val_metrics = {f"val_{k}": v for k, v in run.summary.items() if isinstance(v, (int, float))}
        
        # Extract the weight parameter from the run's config
        config_weight = run.config.get("weight", None)
        
        try:
            # 1. Save config permanently
            config_path = save_run_config(api, entity, project, run.id, base_output_dir)
            
            # 2. Download weights temporarily and run the script
            with download_temp_weights(api, entity, project, run.id) as weight_path:
                cmd = ["bash", "test.sh", "-c", config_path, "-w", weight_path]
                run_dir = os.path.join(base_output_dir, run.id)
                cmd = f"python tools/test.py --config-file {config_path} --num-gpus 1 --options save_path={run_dir} weight={weight_path}"
                print(f"[*] Executing command: {' '.join(cmd)}")
                
                # Execute the bash script. check=True will raise an error if the script fails
                subprocess.run(cmd, check=True)
                print(f"[+] Test script finished successfully for run {run.id}")
            
            # 3. Add to our dataframe
            row_data = {
                "run_id": run.id,
                "run_name": run.name,
                "parent_run_id": parent_run_id,
                "parent_run_name": parent_run_name,
                "lr_split": lr_split,
                "config_weight": config_weight,
                "config_path": config_path,
                "config":dict(run.config)
            }
            row_data.update(val_metrics)
            data.append(row_data)
            
        except subprocess.CalledProcessError as e:
            print(f"[!] Test script failed for run {run.id} with exit code {e.returncode}")
            row_data = {
                "run_id": run.id,
                "run_name": run.name,
                "config_path": config_path,
                "weight_file": "Temporary (Deleted)",
                "parent_run_id": parent_run_id,
                "parent_run_name": parent_run_name,
                "lr_split": lr_split,
                "config_weight": config_weight,
                "test_status": "Failed"
            }
            row_data.update(val_metrics)
            data.append(row_data)
        except Exception as e:
            print(f"[!] Skipping run {run.id} due to error: {e}")
            
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # --- Configuration ---
    WANDB_ENTITY = "pedrosidra"
    WANDB_PROJECT = "diss"
    
    # Build the DataFrame and run the tests inline
    df_experiments = process_and_build_experiments_df(WANDB_ENTITY, WANDB_PROJECT)
    
    print("\n========================================")
    print("Final Experiments DataFrame:")
    print("========================================")
    print(df_experiments.to_string())
    
    csv_path = "experiments_table.csv"
    df_experiments.to_csv(csv_path, index=False)
    print(f"\n[+] Saved DataFrame to {csv_path}")