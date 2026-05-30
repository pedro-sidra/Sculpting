#%%
import pandas as pd


df = pd.read_csv("../experiments_table.csv").fillna("N/A")
#%%
df.loc[df['parent_run_id']=="N/A","parent_run_id"]=df.loc[df['parent_run_id']=="N/A","run_name"].replace(regex=r"[_-]?[lL][rR]\d+",value="")
#%%
order=df.groupby(['parent_run_id']).count().sort_values(by='run_id', ascending=False).index
groups=df.set_index(['parent_run_id', 'lr_split']).loc[order]
groups
# %%
df.pivot_table(index='config_weight', columns='lr_split', values='val_val/mIoU', aggfunc='first')

# pd.crosstab(df, 
#             columns='lr_split',
#              values='val_val/mIoU', aggfunc="mean")
