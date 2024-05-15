#%%
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

path_rounded = '/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-tidy/entropy_in_time_rounded.csv'
path = '/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-tidy/entropy_in_time.csv'

df = pd.read_csv(path)
# %%
# get the first n threads with most comments
def get_first_n_threads(df, n=50):
    df_grouped_comments = df.groupby('submission_id').count().sort_values('final_judg', ascending=False)
    return df_grouped_comments.head(n).index.to_list()


def prepare_df_rounded(path_rounded):
    df_rounded = pd.read_csv(path_rounded)
    df_rounded = df_rounded.T.reset_index()

    #split index between id and judgement using _ as separator 
    df_rounded['id'] = df_rounded['index'].apply(lambda x: x.split('_')[0])
    df_rounded['judg'] = df_rounded['index'].apply(lambda x: x.split('_')[1])
    df_rounded.set_index('id', inplace=True)
    df_rounded.drop(columns=['index'], inplace=True)
    return df_rounded

# given a df of threads plot each of them 
def plot_thread_entropy(df, num_hours=24 ):
    n_rows = len(df) //3
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(10, 50))

    for i, ax in enumerate(axes.flatten()):
        judg = df.iloc[i]['judg']
        data = df.drop(columns=['judg']).iloc[i]
        data.index = data.index.astype(int)
        # Convert index from minutes to hours
        data.index = data.index / 60
        # Plot the data
        sns.lineplot(data=data[:num_hours], ax=ax)
        # Update x ticks labels
        x_ticks = ax.get_xticks()
        #ax.set_xticklabels([f'{int(tick)}:00' for tick in x_ticks])
        ax.set_xlim(0, num_hours)
        #vertical line at 18h
        ax.axline((18, 0), (18,2.7),color='r', linestyle='--')
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Entropy")
        ax.set_title(f'Thread {data.name} ' + judg)
   

    plt.tight_layout()
    plt.show()
# %%

#get the first 20 from rounded 
df_rounded = prepare_df_rounded(path_rounded)
first_50 = get_first_n_threads(df, 50)
first_n_actual = df_rounded.index.intersection(first_50)# avoid missing ids
df_rounded_firsts = df_rounded.loc[first_n_actual]

#delete the columns that are all nan 
#df_rounded_firsts.dropna(axis=1, how='all', inplace=True)




# %%

plot_thread_entropy(df_rounded_firsts,  num_hours=5)
# %%
plot_thread_entropy(df_rounded_firsts, num_hours=24)
# %%
plot_thread_entropy(df_rounded_firsts, num_hours=80)

# %%
