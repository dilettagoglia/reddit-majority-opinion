comments = ["Our city has become so vibrant with all the new cultures and cuisines.",
            "I miss how our town used to be.",
            "It's getting harder to find jobs with all the new people moving in.",
            "I think it's great that we're becoming more multicultural.",
            "I don't feel as safe walking around my neighborhood anymore."
]



#%%
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import xgboost
from utils import * 
import pandas as pd
import seaborn as sns
import numpy as np
import kneed ### https://kneed.readthedocs.io/en/stable/api.html#kneelocator
from datetime import timedelta
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
import igraph as ig
import os
from sklearn.preprocessing import LabelEncoder

#b = Graph()
#df = b.create_graph(in_time=False, t=13)

#p = Prob()
#p.compute_prob(t=00)

#%%
# read a text file
with open('../data/posts_ids.txt') as f:
    posts = [line.strip() for line in f.readlines()]
    


### SENTIMENT

#%%
#t=13

sentiment_before = []
sentiment_after = []
processed_files_path='../data/data-tidy/processed_CSV/'
file_list = [f for f in os.listdir(processed_files_path) if f.endswith('.csv')]
for file_name in file_list:
    file_path = os.path.join(processed_files_path, file_name)
    df = pd.read_csv(file_path, low_memory=False)
    for subm_id, sub_df in df.groupby('submission_id'):
        sub_df['created'] = pd.to_datetime(sub_df['created'], format='%Y-%m-%d %H:%M:%S') # adjust types
        start = sub_df.iloc[0]['created']
        eighteen_h = start + pd.Timedelta(18, 'h', hours=18)
        thread_duration = sub_df.created.max() - sub_df.created.min() 
        if thread_duration < pd.Timedelta(hours=18):
            continue # skip this submission because it does not last enough
        # set the streshold (18h) and split the dataframe
        sub_df_before = sub_df[sub_df['created'] < eighteen_h]
        sub_df_after = sub_df[sub_df['created'] > eighteen_h]
        sub_df_before.reset_index(drop=True, inplace=True)
        sub_df_after.reset_index(drop=True, inplace=True)

        if sub_df_after.empty:
            #print(thread_duration, thread_duration < pd.Timedelta(hours=18))
            continue
        
        sub_df_before = sub_df_before[sub_df_before.text_flair.isna()]  # select no vote comments
        sub_df_after = sub_df_after[sub_df_after.text_flair.isna()]

        if sub_df_after.empty:
            continue

        # decide timerange before and after (not to include the transition period)
        #timerange = pd.Timedelta(t, 'h', hours=t) 
        #end_before = sub_df_before.iloc[-1]['created']
        #start_after = sub_df_after.iloc[0]['created']
        #sub_df_before = sub_df_before[sub_df_before['created'] > (end_before - timerange)]
        #sub_df_after = sub_df_after[sub_df_after['created'] < (start_after + timerange)]

        '''
        sub_df_before['hue'] = 'before'
        sub_df_after['hue'] = 'after'
        prob_together = pd.concat([sub_df_before[['sentiment_compound_VADER', 'hue']], sub_df_after[['sentiment_compound_VADER', 'hue']]])        
        data = pd.melt(prob_together, id_vars=['hue'])
        sns.violinplot(data=data, x='variable', y='value',
                    hue='hue', split=True, gap=.1, inner='box', inner_kws=dict(box_width=10, whis_width=1.2, color="lightgrey"),
                    palette={"before": "r", "after": "g"},
                    legend=None)
        '''
        sentiment_before.append(sub_df_before['sentiment_compound_VADER'].mean())
        sentiment_after.append(sub_df_after['sentiment_compound_VADER'].mean())

differences = np.array(sentiment_after) - np.array(sentiment_before)

plt.figure(figsize=(7, 7))
sns.histplot(differences, kde=True, color='purple')
plt.title('Differences in Means Before and After Event')
plt.xlabel('Difference in Mean')
plt.ylabel('Density')
plt.show()

# Heatmap of Before and After Means
plt.figure(figsize=(14, 7))
sns.heatmap([sentiment_before, sentiment_after], cmap="YlGnBu", cbar=True, xticklabels=False)
plt.title('Heatmap of Means Before and After Event')
plt.ylabel('Before / After')
plt.xlabel('Distribution Index')
plt.show()


#%% TOPICS

df = pd.read_csv('../data/data-tidy/threads_stats.csv')
topics = pd.read_pickle('../data/data-analysis/topics.pkl')
topics.set_index('id', inplace=True)
topics_dict = topics['topic'].to_dict()
df['topic'] = df['subm_id'].map(topics_dict)
df.set_index('subm_id', inplace=True)
#print(df.info())


#%%

df = df.iloc[:, 1:]
df = df.dropna()
X = df[[#'num_comments', 'avg_comm_score', 'post_sentiment'
    'topic', 'assortativity', 'entropy', 'post_score', #'avg_text_len', 
    #'comment_frequency', 'thread_duration', 
    'reciprocity', 'bursts']]
#X.drop(columns=['topic'], inplace=True)
le = LabelEncoder()
df['final_judg'] = le.fit_transform(df['final_judg'])
y = df['final_judg']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
#print(model.summary())

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
xgb = xgboost.XGBClassifier(objective= 'multi:softprob', num_class=4, #objective="reg:squarederror", 
    random_state=16, n_estimators=100, max_depth=3,
                           learning_rate=0.1, verbosity=0)
xgb_model = xgb.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
rfe = RFE(estimator=xgboost.XGBClassifier(), n_features_to_select=5, step=1)

print(f'Predicting target:')
print('TRAIN:')
print(f"XGB MAPE: {metrics.mean_absolute_percentage_error(y_train.values, xgb_model.predict(X_train))}")
r2 = r2_score(y_train.values, xgb_model.predict(X_train))
print(f"XGB R2: {r2}")

print('TEST:')
print(f"XGB MAPE: {metrics.mean_absolute_percentage_error(y_test.values, predictions)}")
r2 = r2_score(y_test.values, predictions)
print(f"XGB R2: {r2}")


#%%
sns.set(font_scale=1.2)
fig, ax = plt.subplots(figsize=(13, 10))
# plt.tight_layout()
data = pd.DataFrame(sorted(zip(xgb.feature_importances_, X_train.columns), reverse=True))
plt.bar(data[1], data[0])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title(f'Feature Importance of XGBoost regressor (R2={str(round(r2, 2))})')
plt.suptitle(f'Predicting {target.name}')
plt.tight_layout()
plt.show()



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
