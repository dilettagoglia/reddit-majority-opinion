import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import entropy
from datetime import timedelta
import seaborn as sns
from sys import exit
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from matplotlib.patches import Patch
from entropy_class import Entropy
import igraph as ig
from math import isnan
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)


def _plot_ecdf(prob_before, prob_after, labels, verdicts, t):
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

def _plot_violin_diff(prob_before, prob_after, labels, verdicts, t):
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
            ax.hlines(0, -0.5, 0.5, colors='r', linestyles='dashed')
            ax.grid(axis='y')
        plt.tight_layout()
        plt.suptitle(f'Final judg = {verdicts[v]} ({len(prob_before[prob_before["final_judg"] == verdicts[v]])} threads)')
        plt.savefig(f'../img/Violin_diff_{str(t)}h__{verdicts[v]}.png')
        plt.show()

       
def _plot_comparison(prob_before, prob_after, labels, verdicts, t, violin=True):
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
            if thread_duration < pd.Timedelta(hours=18+t): # +t removed
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
                    print(thread_duration, thread_duration < pd.Timedelta(hours=18+t)) # +t removed
                    raise ValueError("Empty 'after' dataframe")
                
                # recompute distribution of df after starting the counting from 0 
                sub_df_after[['ESH_perc', 'NAH_perc', 'NTA_perc', 'YTA_perc', 'unsure_perc', 'no_vote_perc']] = sub_df_after.apply(lambda x: pd.Series(e.compute_post_entropy(sub_df_after[sub_df_after.created <= x.created], entropy_param=False)), axis=1)
                #print('Recomputing successful')

                # decide timerange before and after (not to include the transition period)
                # we arbitrarily decided 10 hours before and after the treshold
                #timerange = pd.Timedelta(t, 'h', hours=t) 
                #end_before = sub_df_before.iloc[-1]['created']
                #start_after = sub_df_after.iloc[0]['created']
                #sub_df_before = sub_df_before[sub_df_before['created'] > (end_before - timerange)]
                #sub_df_after = sub_df_after[sub_df_after['created'] < (start_after + timerange)]

                # we take the same number of comments
                #if len(sub_df_after) < 50:
                #    continue
                #sub_df_before = sub_df_before.tail(50)
                #sub_df_after = sub_df_after.head(50)

                sub_df_before.reset_index(drop=True, inplace=True)
                sub_df_after.reset_index(drop=True, inplace=True)

                #print(sub_df_before, sub_df_after)

                final_verdicts_before.append(sub_df_before['final_judg'].values[0])
                final_verdicts_after.append(sub_df_after['final_judg'].values[0])

                # compute probability 
                unique_votes, counts = np.unique(list(sub_df_before['text_flair']), return_counts=True)
                dict_bef = dict(zip(unique_votes, counts / len(sub_df_before['text_flair'])))
                prob_before.append(dict_bef)
                
                unique_votes, counts = np.unique(list(sub_df_after['text_flair']), return_counts=True)
                dict_aft = dict(zip(unique_votes, counts / len(sub_df_after['text_flair'])))
                prob_after.append(dict_aft)
                            
                #print(prob_before, prob_after)

        prob_before=pd.DataFrame(prob_before)
        prob_before['final_judg'] = final_verdicts_before
        prob_after=pd.DataFrame(prob_after)
        prob_after['final_judg'] = final_verdicts_after

        prob_before.fillna(0, inplace=True)
        prob_after.fillna(0, inplace=True)
        print(prob_before, '\n', prob_after)

        prob_before.to_csv(f'../data/data-tidy/prob_before_{str(t)}h.csv', index=False)
        prob_after.to_csv(f'../data/data-tidy/prob_after_{str(t)}h.csv', index=False)
                
        #_plot_ecdf(prob_before, prob_after, labels, verdicts, t)
        _plot_violin_diff(prob_before, prob_after, labels, verdicts, t)
        _plot_comparison(prob_before, prob_after, labels, verdicts, t, violin=True)
        _plot_comparison(prob_before, prob_after, labels, verdicts, t, violin=False)
                    
class Graph:

    def __init__(self):
        self.processed_files_path='../data/data-tidy/processed_CSV/'
    
    def create_graph(self, in_time=False, t=2):
        '''
        file_list = [f for f in os.listdir(self.processed_files_path) if f.endswith('.csv')]
        i=0
        for file_name in file_list:
            i += 1
            file_path = os.path.join(self.processed_files_path, file_name)'''
        dict_a= {}
        dict_e = {}
        dict_a_bef = {}
        dict_a_aft = {}
        file_path = '../data/data-analysis/network-data/user_edgelists.csv'
        df = pd.read_csv(file_path, low_memory=False)
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')

        le = LabelEncoder()
        df.vote = le.fit_transform(df.vote)

        for subm_id, sub_df in df.groupby('root'):
            # build igraph network for each submission 
            #if sub_df.shape[0] < 50:
                #continue

            if in_time: # compute assortativity before and after
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

                    g_bef = ig.Graph.TupleList(sub_df_before[['from', 'to']].itertuples(index=False), directed=True)
                    g_bef.es['created'] = sub_df_before['created'].values
                    g_bef.vs['vote'] = sub_df_before['vote'].values
                    dict_a_bef.update({subm_id: g_bef.assortativity_nominal('vote', directed=True)})
                    dict_a_bef = {k: dict_a_bef[k] for k in dict_a_bef if not isnan(dict_a_bef[k])}

                    g_aft = ig.Graph.TupleList(sub_df_after[['from', 'to']].itertuples(index=False), directed=True)
                    g_aft.es['created'] = sub_df_after['created'].values
                    g_aft.vs['vote'] = sub_df_after['vote'].values
                    dict_a_aft.update({subm_id: g_aft.assortativity_nominal('vote', directed=True)})
                    dict_a_aft = {k: dict_a_aft[k] for k in dict_a_aft if not isnan(dict_a_aft[k])}


            else: 
                g = ig.Graph.TupleList(sub_df[['from', 'to']].itertuples(index=False), directed=True)
                # with nodes ad edges attributes
                g.es['created'] = sub_df['created'].values
                g.vs['vote'] = sub_df['vote'].values

                # compute assortativity 
                assort = g.assortativity_nominal('vote', directed=True)
                dict_a.update({subm_id: assort})

            
        entropy_df = pd.read_csv('../data/data-tidy/entropy_in_time.csv')
        entropy_df['created'] = pd.to_datetime(entropy_df['created'], format='%Y-%m-%d %H:%M:%S')
        for subm_id, sub_df in entropy_df.groupby('submission_id'):
            # update dictionary
            dict_e.update({subm_id: sub_df.entropy_in_time.max()})
        
        '''
        ks_statistic, p_value = stats.ks_2samp(list(dict_a_bef.values()), list(dict_a_aft.values()))
        print(f"KS Statistic: {ks_statistic}")
        print(f"P-value: {p_value}")

        count_bef, bins_bef = np.histogram(list(dict_a_bef.values()), bins=50)
        pdf_bef = count_bef / sum(count_bef)
        cdf_bef = np.cumsum(pdf_bef)
        plt.plot(bins_bef[1:], cdf_bef, color='red', label='Before')

        count_aft, bins_aft = np.histogram(list(dict_a_aft.values()), bins=50)
        pdf_aft = count_aft / sum(count_aft)
        cdf_aft = np.cumsum(pdf_aft)
        plt.plot(bins_aft[1:], cdf_aft, color='blue', label='After')
        plt.xlabel('Assortativity')
        plt.ylabel('ECDF')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(10, 5))
        sns.histplot(list(dict_a_bef.values()), bins=50, kde=True, color='red', label='Before')
        sns.histplot(list(dict_a_aft.values()), bins=50, kde=True, color='blue', label='After')
        plt.xlabel('Assortativity')
        plt.legend()
        plt.show()
        '''

        #print(len(dict_a_bef), len(dict_a_aft))
        
        thread_stats_df = pd.read_csv('../data/data-tidy/threads_stats_struct_prop_rec_and_rand.csv')
        # remove comulns that start with 'Unnamed'
        thread_stats_df = thread_stats_df.loc[:, ~thread_stats_df.columns.str.startswith('Unnamed')]
        #thread_stats_df.set_index('subm_id', inplace=True)

        thread_stats_df['entropy'] = thread_stats_df['subm_id'].map(dict_e)
        thread_stats_df['assortativity'] = thread_stats_df['subm_id'].map(dict_a)
        #thread_stats_df['assort_bef'] = thread_stats_df['subm_id'].map(dict_a_bef)
        #thread_stats_df['assortat_aft'] = thread_stats_df['subm_id'].map(dict_a_aft)
        low_assort = thread_stats_df[thread_stats_df['assortativity'] < 0.5]
        high_assort = thread_stats_df[thread_stats_df['assortativity'] >= 0.5]
        #plt.scatter(thread_stats_df['assort_bef'], thread_stats_df['assortat_aft'], s=0.5, alpha=0.5)
        #plt.yscale('log')
        #plt.xlabel('Before')
        #plt.ylabel('After')
        #plt.loglog()
        #plt.show()

                
        # print correlation coefficient and p value
        print(low_assort.iloc[:, 3:].corr(method='pearson')['assortativity'])
        print(high_assort.iloc[:, 3:].corr(method='pearson')['assortativity'])
        #print(thread_stats_df[['assortativity', 'sentiment', 'entropy']].corr(method='spearman'))
        #print(thread_stats_df[['assortativity', 'sentiment', 'entropy']].corr(method='kendall'))

        # export thread_stats_df as csv
        thread_stats_df.to_csv('../data/data-tidy/threads_stats.csv', index=False)



