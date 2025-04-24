import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import zscore, norm
import numpy as np

labels = ['knowledge','power','status','trust','support','romance','identity','fun','conflict', 'similarity']

def data_prep():

    directory='G:\My Drive\Saved from Chrome'
    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    dfs = [pd.read_pickle(os.path.join(directory, file)) for file in pkl_files]
    merged_df = pd.concat(dfs, ignore_index=False).drop_duplicates()
    merged_df['created'] = pd.to_datetime(merged_df['created'], format='%Y-%m-%d %H:%M:%S')
    merged_df['threshold_col'] = 'A'

    updated_dfs = []
    for subm_id, sub_df in merged_df.groupby('post'):
        start = sub_df['created'].min() 
        threshold = start + pd.Timedelta(hours=18)  
        sub_df.loc[sub_df['created'] < threshold, 'threshold_col'] = 'B' # not working
        updated_dfs.append(sub_df)
        #before = sub_df[sub_df['created'] < threshold]
        #df.drop(before.index, axis=0, inplace=True) # drop them
    merged_df = pd.concat(updated_dfs, ignore_index=False)

    all_coments_judg = pd.read_pickle('./data/data-tidy/all_comments_judg.pkl')
    mapping = pd.read_csv('./data/data-tidy/mapping.csv', index_col=0)
    mapping = dict(zip(mapping.submission_id, mapping.final_judg))
    all_coments_judg['final_judg'] = all_coments_judg['submission_id'].map(mapping)
    all_coments_judg.set_index('id', inplace=True)

    merged_df = merged_df.join(all_coments_judg, how='inner', lsuffix='_l', rsuffix='_r')
    merged_df.final_judg.replace({'Not the A-hole': 'NTA', 'Asshole':'YTA', 'No A-holes here':'NAH', 'Everyone Sucks':'ESH'}, inplace=True)
    merged_df['final_judg'] = merged_df['final_judg'].astype(str)
    merged_df.to_csv('./data/data-analysis/ten_dims_final_4M_comments.csv')

merged_df = pd.read_csv('./data/data-analysis/ten_dims_final_4M_comments.csv')
merged_df.set_index('id', inplace=True, drop=True)
merged_df['created'] = pd.to_datetime(merged_df['created'], format='%Y-%m-%d %H:%M:%S')
merged_df[labels]=merged_df[labels].astype(float)
print(merged_df.describe())
print(merged_df.columns)


###  DEFINE FUNCTIONS 

def balance_data():
    ''' TAKE THE SAME NUMBER OF COMMENTS BEFORE AND AFTER
    '''
    merged_df = merged_df.sort_values(by=['post', 'threshold_col', 'created'], ascending=[True, True, False])
    a_counts = merged_df.groupby('post')['threshold_col'].apply(lambda x: (x == 'A').sum())
    result = []
    for post, count_a in a_counts.items():
        df_post = merged_df[merged_df['post'] == post]
        a_rows = df_post[df_post['threshold_col'] == 'A']
        b_rows = df_post[df_post['threshold_col'] == 'B'].iloc[::-1].head(count_a)
        result.append(a_rows)
        result.append(b_rows)
    merged_df = pd.concat(result).sort_index()
    print(merged_df)

def _plot_distribution(df):
    #merged_df_before = merged_df[merged_df['threshold_col'] == 'B']
    #merged_df_after = merged_df[merged_df['threshold_col'] == 'A']

    df_melted = pd.melt(df, id_vars=['threshold_col'], value_vars=labels, var_name='dimension', value_name='value')
    plt.figure(figsize=(24,12))
    for i, col in enumerate(labels):
        plt.subplot(2, 5, i+1) 
        feature_data = df_melted[df_melted['dimension'] == col]
        #ax = sns.boxplot(x='threshold_col', y='value', data=feature_data, order=['B', 'A'], orient='v', showfliers=False)
        ax = sns.violinplot(x='threshold_col', y='value', data=feature_data, order=['B', 'A'], orient='v', split=True, gap=.1, #inner='box', inner_kws=dict(box_width=10, whis_width=1.2, color="lightgrey"),
                        palette={"B": "r", "A": "g"},
                        legend=None)
        #ax = sns.histplot(data=feature_data, x='value', hue='threshold_col', ax=ax, color='green', kde=True, bins=100,
        #                legend=None, palette={"B": "r", "A": "g"}, stat='count')

        #for j, threshold in enumerate(['B', 'A']):
            #sample_size = merged_df[(merged_df['threshold_col'] == threshold) & (merged_df[col]==1)].shape[0]
            #plt.text(j, df_melted['value'].max() + 0.1, f'n={sample_size}', 
                    #horizontalalignment='center', size=8, color='black')
        ax.set_title(f'{col}')
        ax.set_xlabel(f'{str(len(feature_data[feature_data["value"] > 0]))} data points')

    plt.tight_layout()
    plt.savefig(f'tendim_distribution.png')

def _plot_comparison(df, violin=False):
    df_before = df[df.threshold_col=='B']
    df_after= df[df.threshold_col=='A']
    #df_before['threshold_col'] = 'before'
    #prob_after['hue'] = 'after'
    fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(25, 4))
    for i, ax in enumerate(axes.flatten()):
        #data = df[df['final_judg'] == verdicts[v]]
        #data.fillna(0, inplace=True)
        data = pd.melt(df, id_vars=['threshold_col'], value_vars=[labels[i]])
        if violin:
            sns.violinplot(data=data, x='variable', y='value',
                        hue='threshold_col', split=True, gap=.1, #inner='box', inner_kws=dict(box_width=10, whis_width=1.2, color="lightgrey"),
                        palette={"B": "r", "A": "g"},
                        legend=None, ax=ax)
            ax.set_ylim([-0.19, 1.19])
            ax.grid(axis='y')
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            sns.histplot(data=data, x='value', hue='threshold_col', ax=ax, color='green', kde=True, bins=100, 
                        legend=None, palette={"B": "r", "A": "g"}, stat='count')
            ax.set_xlabel(f'{labels[i]}')
            ax.set_ylabel('')
            ax.set_xlim([0, 1])
    
        if i == 0:
            ax.set_ylabel('Probability') if violin else ax.set_ylabel('Count')

    legend_elements = [Patch(facecolor='r', edgecolor='black', label=f'Before'),
                        Patch(facecolor='g', edgecolor='black', label=f'After')]
    fig.legend(handles=legend_elements, loc='upper right', ncol=len(labels))

    plt.tight_layout()
    #plt.suptitle(f'Final judg = {verdicts[v]} ({len(prob_before[prob_before["final_judg"] == verdicts[v]])} threads)')
    title_str = 'Violin_compar' if violin else 'Distr'
    plt.savefig(f'{title_str}.png')
    plt.show()    

def compute_odd_ratios(merged_df, strategy='by_verdict', majority=''):
    '''
    options for 'strategy' param:
        - before_after
        - by_verdict
    '''
    odds_ratios = []
   
    if strategy == 'before_after':
        for col in labels:
            ct = pd.crosstab(merged_df['threshold_col'], merged_df[col]) # contingency table: group vs feature
            # ensure all values exist (fill with 0)
            a = ct.at['A', 1] if 1 in ct.columns and 'A' in ct.index else 0 #b
            b = ct.at['A', 0] if 0 in ct.columns and 'A' in ct.index else 0 #d
            c = ct.at['B', 1] if 1 in ct.columns and 'B' in ct.index else 0 #a
            d = ct.at['B', 0] if 0 in ct.columns and 'B' in ct.index else 0 #c

            if b == 0 or c == 0:
                or_value = float('inf') if a > 0 and d > 0 else 0 # avoid divide by zero
            else:
                or_value = (a * d) / (b * c)
            #odds_ratios[col] = or_value # if odds_ratios is a dict

            # Add 0.5 to all cells to handle zeros (Haldane-Anscombe correction)
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5

            log_or = np.log(or_value)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_low = np.exp(log_or - z * se)
            ci_high = np.exp(log_or + z * se)
            odds_ratios.append({
                'feature': col,
                'odds_ratio': or_value,
                'ci_5': ci_low,
                'ci_95': ci_high
            })
    
    elif strategy == 'by_verdict':
        if majority in ['NTA', 'YTA', 'NAH', 'ESH', 'ALL']:
            merged_df = merged_df[merged_df.threshold_col == 'A'] # only after
            #merged_df = merged_df[merged_df.final_judg == majority] # one computation per verdict (select that verdict) # TODO ripartire da qui
            #merged_df['agree'] = merged_df['text_flair'].apply(lambda x: 'Y' if x == majority else 'N')

            merged_df['agree'] = merged_df.apply(lambda row: 'Y' if row['text_flair_r'] == row['final_judg'] else 'N', axis=1)

            for col in labels:
                ct = pd.crosstab(merged_df['agree'], merged_df[col])
                a = ct.at['Y', 1] if 1 in ct.columns and 'Y' in ct.index else 0 
                b = ct.at['Y', 0] if 0 in ct.columns and 'Y' in ct.index else 0 
                c = ct.at['N', 1] if 1 in ct.columns and 'N' in ct.index else 0 
                d = ct.at['N', 0] if 0 in ct.columns and 'N' in ct.index else 0 
                
                if b == 0 or c == 0:
                    or_value = float('inf') if a > 0 and d > 0 else 0 # avoid divide by zero
                else:
                    or_value = (a * d) / (b * c)
                #odds_ratios[col] = or_value

                a += 0.5
                b += 0.5
                c += 0.5
                d += 0.5

                log_or = np.log(or_value)
                se = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_low = np.exp(log_or - z * se)
                ci_high = np.exp(log_or + z * se)

                odds_ratios.append({
                    'feature': col,
                    'odds_ratio': or_value,
                    'ci_5': ci_low,
                    'ci_95': ci_high
                })

        else:
            raise ValueError
            print('Select a valid majority param value to compute odds ratio')

    else:
        raise ValueError 
        print('Select a valid strategy param to compute odds ratio')

    
    return odds_ratios



### APPLY

# thresholding 80%
merged_df[labels] = (merged_df[labels] > 0.7).astype(int)
z = norm.ppf(0.95)  

balance_data()
#_plot_distribution(merged_df) # boxplot
#_plot_comparison(merged_df, violin=True) # violin plot

majority_param='ALL'
odds_ratios = compute_odd_ratios(merged_df, strategy='by_verdict', majority=majority_param)

#df_or = pd.DataFrame(list(odds_ratios.items()), columns=['feature', 'odds_ratio'])
df_or = pd.DataFrame(odds_ratios)

p_values = []
for i, row in df_or.iterrows():
    se = np.log(row['ci_95']) - np.log(row['ci_5']) / (2 * z) # standard error
    log_or = np.log(row['odds_ratio'])
    z_value = log_or / se
    p_value = 2 * (1 - norm.cdf(abs(z_value)))  # two-tailed test
    p_values.append(p_value)
df_or['p_value'] = p_values
print(df_or)

plt.figure(figsize=(8, 4))
sns.set_style("whitegrid")
plt.axvline(1, color='gray', linestyle='--', linewidth=1)
for i, row in df_or.iterrows():
    plt.plot([row['odds_ratio']], [row['feature']], 'o', color='steelblue', markerfacecolor='white', markeredgewidth=2)
    plt.plot([row['ci_5'], row['ci_95']], [i, i], color='black', lw=2)  # CI line



x_min, x_max = plt.xlim()
x_mid = 1  
y_pos = -1.8    
plt.text((x_min + x_mid) / 2, y_pos, 'Disagree', ha='center', va='top', fontsize=10, fontweight='bold')
plt.text((x_max + x_mid) / 2, y_pos, 'Agree', ha='center', va='top', fontsize=10, fontweight='bold')
plt.ylim(-1, len(df_or) - 0.5) # Stretch y-limits to give space for the bottom labels

plt.xlabel('Odds Ratio')
plt.ylabel('Dimension')
plt.title(f'{majority_param} odds ratios AFTER the verdict')
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
#plt.xscale('log')  # Optional: Log scale for symmetry if you have both OR >1 and OR <1
plt.savefig(f'odd_ratios_{majority_param}_balanced.png')





