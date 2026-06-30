#%%
import pandas as pd
from pathlib import Path
import json


df = pd.read_csv("../experiments_table.csv").fillna("N/A")
#%%
no_split=df.loc[df['lr_split']=='N/A']
df.loc[no_split.index,'lr_split'] = no_split['run_name'].str.upper().str.extract(r"(LR\d+)").values
#%%
df.loc[df['parent_run_id']=="N/A","parent_run_name"]=df.loc[df['parent_run_id']=="N/A","run_name"].replace(regex=r"[_-]?[lL][rR]\d+",value="")
#%%
df['results_file'] = "../run_data/"+df['run_id'] +"/results.json"
results=pd.json_normalize(df['results_file'].apply(lambda x: None if not Path(x).is_file() else json.load(Path(x).open())))
df=pd.concat((df,results),axis=1)
#%%
order=df.groupby(['parent_run_name']).count().sort_values(by='run_id', ascending=False).index
groups=df.set_index(['parent_run_name', 'lr_split']).loc[order]
groups
#%%
df.pivot_table(
    index='parent_run_name',
    columns="lr_split",
    values='mIoU'
).to_csv("results.csv")

# %%
