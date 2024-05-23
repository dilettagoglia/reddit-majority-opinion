#%%
from matplotlib import pyplot as plt
from utils import * 
import pandas as pd
import seaborn as sns
import numpy as np
import kneed ### https://kneed.readthedocs.io/en/stable/api.html#kneelocator
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
import igraph as ig

b = Graph()
df = b.create_graph()


#%%

e = Entropy()

df = e.user_nodelist()
df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S') # adjust types
knees = []
variance_list_pre_18 = []
variance_list_post_18 = []

for subm_id, sub_df in df.groupby('submission_id'):
    sub_df.sort_values(by='created', inplace=True)

    # round to 1 minute
    sub_df['time_rounded'] = e.minute_rounder(sub_df.created)
    sub_df['time_int'] = sub_df['time_rounded'].diff().fillna(timedelta(0)).apply(lambda x: x.total_seconds() / 60)
    sub_df['time_int'] = sub_df['time_int'].cumsum()
    sub_df = sub_df.groupby(['time_rounded']).max() 
    sub_df.set_index('time_int', inplace=True) 
    sub_df['time_int'] = sub_df.index
    sub_df.reset_index(inplace=True, drop=True)
    
    # find elbow
    kneedle = kneed.KneeLocator(x=sub_df.time_int, y=sub_df.entropy_in_time, curve="concave", direction="increasing", S=100)
    knee_point = kneedle.knee   
    #kneedle.plot_knee()
    #plt.show()
    knees.append(knee_point)

    '''
    # compute variance elbow-->18h and 18h-->end
    start = sub_df['time_int'].iloc[0]  # df is already sorted by time ascending
    eighteen_h = start + 18*60 # 18 hours in minutes
    #print(start, knee_point, eighteen_h)
    variance_list_pre_18.append(sub_df[(sub_df['time_int'] > knee_point) & (sub_df['time_int'] < eighteen_h)].entropy_in_time.std())
    variance_list_post_18.append(sub_df[sub_df['time_int'] >= eighteen_h].entropy_in_time.std())
    '''
knees = [k for k in knees if k is not None]
print(len(knees), min(knees), max(knees), np.mean(knees), np.std(knees))
# 2932 0.0 24959.0 334.38369713506137 961.056692933871

#sns.histplot(variance_list_pre_18, bins=30, kde=True)
#sns.histplot(variance_list_post_18, bins=30, kde=True)
#plt.show()

exit(0)




#%%
df = e.user_nodelist()
df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f') # adjust types
df['duration'] = df.groupby('submission_id')['created'].transform(lambda x: x.max() - x.min()) # compute duration for each submission (group): created last - created first
print(df)
#histogram pf duration
sns.histplot(df['duration'].dt.days, bins=30, kde=True)
plt.show()

# %%
# histogram of df length grouped by submission_id
bins = 30
counts, bins, _ = plt.hist(df.groupby('submission_id').size(), bins=bins)
plt.bar_label(_)
print(bins[0]) # how many in the first bin
plt.show()

#%%
bins = 10
# filter length less than 2000
df_short_networks = df.groupby('submission_id').filter(lambda x: len(x) < 2000)
#counts, bins, _ = plt.hist(df_short_networks.groupby('submission_id').size(), bins=bins)
# same but woth sns
sns.histplot(df_short_networks.groupby('submission_id').size(), bins=bins, kde=True)
plt.bar_label(_)
print(bins) # how many in the first bin
plt.xlim(0, 500)
plt.show()
# %%
