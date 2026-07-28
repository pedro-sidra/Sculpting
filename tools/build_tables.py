#%%
import pandas as pd
from pathlib import Path
import json
import wandb

# --- W&B API CONFIGURATION ---
ENTITY = "pedrosidra"   # Replace with your W&B username or team name
PROJECT = "diss" # Replace with your W&B project name

df = pd.read_csv("../experiments_table.csv").fillna("N/A")

#%%
no_split = df.loc[df['lr_split'] == 'N/A']
df.loc[no_split.index, 'lr_split'] = no_split['run_name'].str.upper().str.extract(r"(LR\d+)").values

#%%
df.loc[df['parent_run_id'] == "N/A", "parent_run_name"] = df.loc[df['parent_run_id'] == "N/A", "run_name"].replace(regex=r"[_-]?[lL][rR]\d+", value="")

#%%
df['results_file'] = "../run_data/" + df['run_id'] + "/results.json"
results = pd.json_normalize(df['results_file'].apply(lambda x: None if not Path(x).is_file() else json.load(Path(x).open())))
df = pd.concat((df, results), axis=1)

#%%
# --- FETCH DATES FROM WANDB API ---
api = wandb.Api()
project_runs = api.runs(f"{ENTITY}/{PROJECT}")

# Create dictionary. Local runs not in this dict will safely become 'NaT' (Not a Time)
run_dates = {run.id: pd.to_datetime(run.created_at) for run in project_runs}

df['created_at'] = df['run_id'].map(run_dates)
# ----------------------------------

#%%
order = df.groupby(['parent_run_name']).count().sort_values(by='run_id', ascending=False).index
groups = df.set_index(['parent_run_name', 'lr_split']).loc[order]
groups

#%%
table = df.pivot_table(
    index='parent_run_name',
    columns="lr_split",
    values='mIoU'
)

# --- SORTING LOGIC WITH LOCAL RUNS FIRST & INDEX SAFEGUARD ---
# max() ignores NaT by default if there's a mix of W&B and local runs in a group.
# na_position='first' forces parent_run_names that are ENTIRELY local (NaT) to the top.
sorted_dates = df.groupby('parent_run_name')['created_at'].max().sort_values(
    ascending=True, 
    na_position='first'
)

# Filter sorted_dates to ONLY include parent_run_names that exist in the pivot table
valid_sorted_index = sorted_dates.index[sorted_dates.index.isin(table.index)]

# Reorder the pivoted table safely
table = table.loc[valid_sorted_index]
# -------------------------------------------------------------

breakpoint()
table.to_csv("results.csv")

# %%