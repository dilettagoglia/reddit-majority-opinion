#%%
import pandas as pd
import os

processed_files_path='/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-tidy/processed_CSV'



final_nodelist = pd.DataFrame()
file_list = [f for f in os.listdir(processed_files_path) if f.endswith('.csv')]
for file_name in file_list:
    file_path = os.path.join(processed_files_path, file_name)
    df = pd.read_csv(file_path, low_memory=False)
   
    df = df[df['type'] =='sub']

    final_nodelist = pd.concat([final_nodelist, df], ignore_index=True)



# %%

#rename columns
posts = final_nodelist.rename(columns={'title':'text'})

# remove all words that are  AITA , YTA, NTA, ESH, NAH
posts['text'] = posts['text'].str.replace('AITA', '')
posts['text'] = posts['text'].str.replace('YTA', '')
posts['text'] = posts['text'].str.replace('NTA', '')
posts['text'] = posts['text'].str.replace('ESH', '')
posts['text'] = posts['text'].str.replace('NAH', '')



cache ='/Users/alessiogandelli/dev/uni/reddit-disagreement/data/cache/'

# %%
from tweets_to_topic_network.topic import Topic_modeler

tm = Topic_modeler(posts, name = 'AITA', embedder_name='text-embedding-3-large', path_cache = cache)

# %%
df_labeled = tm.get_topics()


df_labeled[df_labeled['topic'] == 13]['text'].values
# %%
