import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from datetime import timedelta
import seaborn as sns
from sys import exit
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from matplotlib.patches import Patch
from entropy_class import Entropy

def plot_ecdf(prob_before, prob_after, labels, verdicts, t):
    for v in range(len(verdicts)):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5), sharex=True)

        for i, ax in enumerate(axes.flatten()):

            # In order to plot the histogram we need to transform voting labels into numbers (we use LabelEncoder)
            # TODO Note that here we are note distinguishing between people that did not vote and people that voted unsure
                # to fix this, see the function user_nodelist_per_post()

            le = LabelEncoder()
            prob_before[labels[i]] = le.fit_transform(prob_before[labels[i]])
            prob_after[labels[i]] = le.fit_transform(prob_after[labels[i]])

            # TODO normalize x axis (thread size)

            count, bins_count = np.histogram(prob_before[prob_before['final_judg'] == verdicts[v]][labels[i]], bins=10)
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            ax.plot(bins_count[1:], cdf, color='red', label='Before 18h')

            count, bins_count = np.histogram(prob_after[prob_after['final_judg'] == verdicts[v]][labels[i]], bins=10)
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            ax.plot(bins_count[1:], cdf, color='green', label='After 18h')

            ax.set_xlabel(f'{labels[i]}')

        plt.tight_layout()
        plt.suptitle(f'Final judg = {verdicts[v]} ({len(prob_before[prob_before["final_judg"] == verdicts[v]])} threads)')
        #plt.savefig(f'../img/ECDF__{verdicts[v]}.png')
        plt.show()

def plot_violin_diff(prob_before, prob_after, labels, verdicts, t):
    for v in range(len(verdicts)):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
        for i, ax in enumerate(axes.flatten()):
            sns.violinplot(prob_after[prob_after['final_judg'] == verdicts[v]][labels[i]] - prob_before[prob_before['final_judg'] == verdicts[v]][labels[i]],  
                            ax=ax, color='blue')
            ax.set_xlabel(f'{labels[i]}')
            ax.set_ylabel('')
            ax.set_ylim([-1, 1])
            if i == 0:
                ax.set_ylabel('Difference of probabilities (after - before)')

        plt.tight_layout()
        plt.suptitle(f'Final judg = {verdicts[v]} ({len(prob_before[prob_before["final_judg"] == verdicts[v]])} threads)')
        plt.savefig(f'../img/Violin_diff_{str(t)}h__{verdicts[v]}.png')
        plt.show()

def plot_comparison(prob_before, prob_after, labels, verdicts, t, violin=True):
    prob_before['hue'] = 'before'
    prob_after['hue'] = 'after'
    prob_together = pd.concat([prob_before, prob_after])

    for v in range(len(verdicts)):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))

        for i, ax in enumerate(axes.flatten()):
            data = prob_together[prob_together['final_judg'] == verdicts[v]]
            data.fillna(0, inplace=True)
            data = pd.melt(data, id_vars=['final_judg', 'hue'], value_vars=[labels[i]])
            if violin:
                sns.violinplot(data=data, x='variable', y='value',
                            hue='hue', split=True, gap=.1, inner='box', inner_kws=dict(box_width=10, whis_width=1.2, color="lightgrey"),
                            palette={"before": "r", "after": "g"},
                            legend=None, ax=ax)
                ax.set_ylim([-0.19, 1.19])
                ax.grid(axis='y')
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                sns.histplot(data=data, x='value', hue='hue', ax=ax, color='green', kde=True, bins=10, 
                            legend=None, palette={"before": "r", "after": "g"}, stat='count')
                ax.set_xlabel(f'{labels[i]}')
                ax.set_ylabel('')
                ax.set_xlim([0, 1])
        
            if i == 0:
                ax.set_ylabel('Probability') if violin else ax.set_ylabel('Count')

        legend_elements = [Patch(facecolor='r', edgecolor='black', label=f'{str(t)}h Before'),
                            Patch(facecolor='g', edgecolor='black', label=f'{str(t)}h After')]
        fig.legend(handles=legend_elements, loc='upper right', ncol=len(labels))

        plt.tight_layout()
        plt.suptitle(f'Final judg = {verdicts[v]} ({len(prob_before[prob_before["final_judg"] == verdicts[v]])} threads)')
        title_str = 'Violin_compar' if violin else 'Distr'
        plt.savefig(f'../img/{title_str}_{str(t)}h__{verdicts[v]}.png')
        plt.show()    

class Preprocess:
    def __init__(self):
        self.processed_files_path='../data/data-tidy/processed_CSV/'

    def preprocess_votes(self):
        file_list = [f for f in os.listdir(self.processed_files_path) if f.endswith('.csv')]
        i=0
        for file_name in file_list:
            i += 1
            file_path = os.path.join(self.processed_files_path, file_name)
            df = pd.read_csv(file_path, low_memory=False)
            df['number_of_votes'] = df['text_flair_list'].str.split(',').str.len() # no vote and 1 vote have both value 1 (because no comma)
            df.loc[(df.number_of_votes > 1) & (df.text_flair.isna()), 'text_flair'] = 'unsure'
            #print(df[df.number_of_votes > 1][['text_flair', 'text_flair_list', 'number_of_votes']])
            # replace NaN with empty string
            df['text_flair'] = df['text_flair'].fillna('')
            df.to_csv(file_path, index=False)
            print(f'Processed file {i}/{len(file_list)}')      

class Prob:
    def __init__(self):
        # vars
        self.processed_files_path='../data/data-tidy/processed_CSV/'
        self.user_nodelist_export = '../data/data-tidy/user_nodelist_per_post.csv'
    
    def create_user_nodelist_per_post(self):
        final_nodelist = pd.DataFrame()
        file_list = [f for f in os.listdir(self.processed_files_path) if f.endswith('.csv')]
        for file_name in file_list:
            file_path = os.path.join(self.processed_files_path, file_name)
            df = pd.read_csv(file_path, low_memory=False)
            df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S') # adjust types
            for subm_id, sub_df in df.groupby('submission_id'):
                sub_df.text_flair = sub_df.text_flair.fillna('unsure')
                node_df=pd.DataFrame()
                node_df[['author', 'entering_time']] = sub_df[['created', 'author']].groupby('author').min().reset_index()
                node_df['post'] = subm_id
                node_df.set_index('author', inplace=True)
                # create a column that is list of flair of all authors
                node_df['flairs'] = sub_df.groupby('author')['text_flair'].apply(", ".join)
                node_df.reset_index(inplace=True)
                final_nodelist = pd.concat([node_df, final_nodelist])
                final_nodelist.reset_index(drop=True, inplace=True)
        final_nodelist.to_csv(self.user_nodelist_export, index=False)
    
    def compute_prob(self, t=2):
        
        labels = ['ESH', 'NAH', 'NTA', 'YTA', 'unsure', '']
        verdicts = ['Not the A-hole', 'Asshole', 'No A-holes here', 'Everyone Sucks']

        e = Entropy()
        df = e.user_nodelist(entropy_param=False) # import perc_disagreement_in_time 
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')
        df.text_flair.fillna('', inplace=True) # no votes
        prob_before=[]
        prob_after=[]
        final_verdicts_before = []
        final_verdicts_after = []
        for subm_id, sub_df in df.groupby('submission_id'):
            # filter out threads that last less than 18+t hours
            thread_duration = sub_df.created.max() - sub_df.created.min() 
            if thread_duration < pd.Timedelta(hours=18+t):
                continue # skip this submission because it does not last enough
            else:                
                sub_df.sort_values(by='created', inplace=True)
                start = sub_df.iloc[0]['created']
                eighteen_h = start + pd.Timedelta(18, 'h', hours=18)

                # remove the rows with no votes
                # sub_df = sub_df[~sub_df['text_flair'].isna()]
                # TEMPORARLY REMOVED> we also consider them in the plot

                # set the streshold (18h) and split the dataframe
                sub_df_before = sub_df[sub_df['created'] < eighteen_h]
                sub_df_after = sub_df[sub_df['created'] > eighteen_h]
                sub_df_before.reset_index(drop=True, inplace=True)
                sub_df_after.reset_index(drop=True, inplace=True)

                if sub_df_after.empty:
                    print(thread_duration, thread_duration < pd.Timedelta(hours=18+t))
                    raise ValueError("Empty 'after' dataframe")

                # decide timerange before and after (not to include the transition period)
                # we arbitrarily decided 10 hours before and after the treshold
                timerange = pd.Timedelta(t, 'h', hours=t) 
                end_before = sub_df_before.iloc[-1]['created']
                start_after = sub_df_after.iloc[0]['created']
                sub_df_before = sub_df_before[sub_df_before['created'] > (end_before - timerange)]
                sub_df_after = sub_df_after[sub_df_after['created'] < (start_after + timerange)]

                final_verdicts_before.append(sub_df_before['final_judg'].values[0])
                final_verdicts_after.append(sub_df_after['final_judg'].values[0])

                # compute probability 
                unique_votes, counts = np.unique(list(sub_df_before['text_flair']), return_counts=True)
                prob_before.append(counts / len(sub_df_before['text_flair']))
                #prob_before.append(counts)
                unique_votes, counts = np.unique(list(sub_df_after['text_flair']), return_counts=True)
                prob_after.append(counts / len(sub_df_after['text_flair']))
                #prob_after.append(counts)

        prob_before=pd.DataFrame(prob_before, columns=[labels])
        prob_before['final_judg'] = final_verdicts_before
        prob_after=pd.DataFrame(prob_after, columns=[labels])
        prob_after['final_judg'] = final_verdicts_after

        # fix multindex
        prob_before.columns = [col[0] for col in prob_before.columns] 
        prob_after.columns = [col[0] for col in prob_after.columns]
                
        #plot_ecdf(prob_before, prob_after, labels, verdicts, t)
        plot_violin_diff(prob_before, prob_after, labels, verdicts, t)
        #plot_comparison(prob_before, prob_after, labels, verdicts, t, violin=True)
                    




