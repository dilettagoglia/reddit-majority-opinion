#%%
from matplotlib import pyplot as plt
from utils import Entropy
import pandas as pd
import seaborn as sns

e = Entropy()

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
