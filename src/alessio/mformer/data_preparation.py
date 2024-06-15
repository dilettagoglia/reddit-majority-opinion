#%%
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import os
import torch

# %%

def get_posts_ids(path_posts):
    with open(path_posts) as f:
        posts = [line.strip() for line in f.readlines()]
    return posts

path_posts = '/Users/alessiogandelli/dev/uni/reddit-disagreement/data/posts_ids.txt'


#%%

post_ids = get_posts_ids(path_posts)
processed_files_path='/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-tidy/processed_CSV'


# we have to search for the posts in this list of files 
file_list = [f for f in os.listdir(processed_files_path) if f.endswith('.csv')]

df = pd.DataFrame()

for file_name in file_list:
    file_path = os.path.join(processed_files_path, file_name)
    df1 = pd.read_csv(file_path, low_memory=False)
    df = pd.concat([df, df1], ignore_index=True)

df.to_pickle(processed_files_path+'all_merged.pkl')

#%%


# we need the comments ids for each post keys are the post ids and values are the list of comments ids
comments_id = {} 

for post in post_ids:
    comments_id[post] = df[df['submission_id'] == post]['id'].to_list()
    

   


#%%

comments_files_path='/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-raw'



comments_file_list = [os.path.join(comments_files_path,f) for f in os.listdir(comments_files_path) if f.endswith('.csv')]


#%%
df_comments = pd.DataFrame()
for file in comments_file_list:
    df1 = pd.read_csv(file, low_memory=False)
    df_comments = pd.concat([df_comments, df1], ignore_index=True)

df_comments.to_pickle('all_comments_merged.pkl')

#%%

df_comments_id = pd.DataFrame(list(comments_id.items()), columns=['post', 'comment_id'])

# Convert the 'comment_id' column from lists to individual rows
df_comments_id = df_comments_id.explode('comment_id')
df_merged = pd.merge(df_comments, df_comments_id, left_on='id', right_on='comment_id', how='inner')

df_merged.to_pickle('all_comments_merged.pkl')

# this is a table with comment id, post id, text




# %%
