import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from datetime import timedelta

class Entropy:
    def __init__(self):
        # vars
        self.raw_files_path='../data/data-raw/CSV'
        self.processed_files_path='../data/data-tidy/processed_CSV'
        self.entropy_export = "../data/data-tidy/entropy_in_time.csv"
        self.entropy_in_time_df = None

    def compute_post_entropy(self, df):
        df = df[df.voter==1]
        unique_votes, counts = np.unique(list(df.text_flair), return_counts=True)
        prob = counts / len(df.text_flair)
        entr = round(entropy(prob, base=2), 2)
        return entr

    def compute_entropy_in_time(self, sub_df, final_judg):
        sub_df.sort_values(by='created', inplace=True)
        #df = sub_df
        df_star = sub_df[sub_df.depth<1]
        df_per = sub_df[sub_df.depth>=1]
        df_star['subgraph'] = 'star'
        df_per['subgraph'] = 'per'
        df_star['entropy_in_time'] = df_star.apply(lambda x: self.compute_post_entropy(df_star[df_star.created <= x.created]), axis=1)
        df_per['entropy_in_time'] = df_per.apply(lambda x: self.compute_post_entropy(df_per[df_per.created <= x.created]), axis=1)
        df_star['final_judg'] = final_judg
        df_per['final_judg'] = final_judg
        df=pd.concat([df_star, df_per])
        return df[['submission_id', 'final_judg', 'created', 'entropy_in_time', 'subgraph']]

    def user_nodelist(self):
        if os.path.exist(self.entropy_export):
            self.entropy_in_time_df = pd.read_csv(self.entropy_export)
        entropy_df_merged=pd.DataFrame()
        file_list = [f for f in os.listdir(self.processed_files_path) if f.endswith('.csv')]
        pbar = tqdm(total=6366)
        for file_name in file_list:
            file_path = os.path.join(self.processed_files_path, file_name)
            df = pd.read_csv(file_path, low_memory=False)
            df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f') # adjust types

            for subm_id, sub_df in df.groupby('submission_id'): # groupby excludes NaNs so OP are not counted in the authors list
                if sub_df.type.nunique() != 1:
                    print('ERROR: in submissions considered', file_name, subm_id)
                    raise ValueError

                final_judg=df[df.id==subm_id].link_flair_text.values[0]
                entropy_df = self.compute_entropy_in_time(sub_df, final_judg)
                entropy_df_merged = pd.concat([entropy_df_merged, entropy_df])
                pbar.update(n=1)
        entropy_df_merged.to_csv(self.entropy_export, index=False)  
        self.entropy_in_time_df = entropy_df_merged
    
    def minute_rounder(self, t):
        '''
        Rounds to nearest minute by adding a timedelta, minute if second >= 30.

        :param t: datetime object
        :return: datetime object rounded to nearest minute
        '''

        return (t.map(lambda x : x.replace(second=0, microsecond=0, minute=x.minute, hour=x.hour)
                +timedelta(minutes=x.second//30)))

    def ten_minute_rounder(self, dt):
        return (dt.map(lambda x : x.replace(second=0, microsecond=0, minute=x.minute//10*10, hour=x.hour)))#

    def hour_rounder(self, t):
        return (t.map(lambda x : x.replace(second=0, microsecond=0, minute=0, hour=x.hour)
                +timedelta(hours=x.minute//30)))
        
    def entropy_in_time_plot(self):
        df = pd.read_csv(self.entropy_export, low_memory=False)
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f') # adjust types
        print(df.final_judg.unique())
        df_avg_entropy_before={}
        temp = []

        for subm_id, sub_df in df.groupby('submission_id'):
            sub_df.reset_index(drop=True, inplace=True)

            # COMPUTE THE 18H THRESHOLD
            start = sub_df['created'].iloc[0]  # df is already sorted by time ascending
            eighteen_h = start + pd.Timedelta(18, 'h', hours=18)
            sub_df.set_index('created', inplace=False)

            try:
                temp.append(sub_df[sub_df['created'] > eighteen_h].index[0])
            except IndexError:
                continue
            #try:
                #thresh = sub_df[sub_df['created'] > eighteen_h].index.to_list()[0]  # first timestamp over 18h
            #except IndexError:
                #thresh = sub_df.index[-1]  # if no timestamp over 18h, then take the last timestamp
                #continue # skip this submission

            sub_df['number'] = sub_df.index
            sub_df['time_rounded'] = self.minute_rounder(sub_df.created)
            sub_df['time_int'] = sub_df['time_rounded'].diff().fillna(timedelta(0)).apply(
                lambda x: x.total_seconds() / 60)
            sub_df['time_int'] = sub_df['time_int'].cumsum()
            #sub_df = sub_df.groupby(['time_rounded']).max() # TEMPORARY REMOVED
            #sub_df.set_index('time_int', inplace=True) # TEMPORARY REMOVED

            col_name =f'{subm_id}_{str(sub_df.final_judg.unique())}'
            serie = sub_df['entropy_in_time']

            #serie_2 = sub_df['entropy_in_time'][thresh:]
            #serie_2.reset_index(drop=True, inplace=True)
            df_avg_entropy_before.update({col_name: serie})

        df_avg_entropy_before = pd.DataFrame(df_avg_entropy_before)
        #df_avg_entropy_before['mean_at_timestamp'] = df_avg_entropy_before.mean(axis=0)

        df_avg_entropy_before = df_avg_entropy_before[df_avg_entropy_before.index < 8640]

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5), dpi=600)
        ax1.set_xticks(np.arange(0, 8641, 1440))
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        labels = [str(int(int(label)/60)) for label in labels]
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_xlabel('Hours', fontsize=12)
        ax1.set_ylabel('Average Post Entropy', fontsize=12)
        #ax1.set_title('Before 18h')
        ax1.set_yticks(np.arange(0, 2.6, 0.4))
        ax1.legend()

        # threshold after 18 hours (vertical line)
        plt.axvline(x=1080, color='green', linewidth=1, linestyle='dashed')
        plt.axvline(x=1440, color='grey', linewidth=0.5, linestyle='dotted')
        plt.axvline(x=2880, color='grey', linewidth=0.5, linestyle='dotted')
        plt.axvline(x=4320, color='grey', linewidth=0.5, linestyle='dotted') # day 3
        plt.axvline(x=5760, color='grey', linewidth=0.5, linestyle='dotted')
        plt.axvline(x=7200, color='grey', linewidth=0.5, linestyle='dotted') # day 5
        plt.grid(axis='y') # add only horizontal grid
        plt.show()
        #plt.savefig('../data-analysis/paper_figs/entropy_in_time_by_final_judg.png', dpi=600)
