#%%
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

path = '/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-tidy/user_nodelist_per_post.csv'

df = pd.read_csv(path)
# %%
# if flairs is nan remove line 
df = df.dropna(subset=['flairs'])
df['flairs'] = df['flairs'].astype(str)

# %%
# %%
df.loc[:, 'n_comments'] = df['flairs'].apply(lambda x: len(x.split(',')))

df.loc[:, 'n_votes'] = df['flairs'].apply(lambda x: len(set([i for i in x.split(',') if i.strip() != ''])))# %%

df.loc[:, 'NTA_votes'] = df['flairs'].apply(lambda x: len([i for i in x.split(',') if i.strip() == 'NTA']))
df.loc[:, 'YTA_votes'] = df['flairs'].apply(lambda x: len([i for i in x.split(',') if i.strip() == 'YTA']))
df.loc[:, 'ESH_votes'] = df['flairs'].apply(lambda x: len([i for i in x.split(',') if i.strip() == 'ESH']))
df.loc[:, 'NAH_votes'] = df['flairs'].apply(lambda x: len([i for i in x.split(',') if i.strip() == 'NAH']))

# %%
df[df['n_comments'] > 1].sort_values('n_comments', ascending=False)
# %%
df[df['n_votes'] > 1].sort_values('n_votes', ascending=False)
# %%
# i have this 4 columns NTA_votes	YTA_votes	ESH_votes	NAH_votes, i wamt to find the rows in which there are at least 2 of them  > 0

df

# %%
df[df['n_votes'] > 2].groupby('post').count().sort_values('flairs', ascending=False)
# %% #look for discordant votes 

df['vote_types'] = (df[['NTA_votes', 'YTA_votes', 'ESH_votes', 'NAH_votes']] > 0).sum(axis=1)
df_filtered = df[df['vote_types'] >= 2]
# %%
df_grouped = df.groupby('post').agg({'author': ['count']})

# %%
