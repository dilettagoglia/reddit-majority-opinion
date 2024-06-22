#%%
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import os

# Change the foundation name if need be 
FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "sanctity"]

MODEL_NAME = f"joshnguyen/mformer-"

# Load model and tokenizer
models = {}
for foundation in FOUNDATIONS:
    tokenizer = AutoTokenizer.from_pretrained(  MODEL_NAME + foundation)
    models[foundation] = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME + foundation,
        device_map="auto"
    )

#%%
df = pd.read_pickle('C:/Users/dilgo529/Documents/GitHub/reddit-disagreement/data/data-tidy/all_comments_merged.pkl')
df_2 = pd.read_pickle('C:/Users/dilgo529/Documents/GitHub/reddit-disagreement/data/data-tidy/all_comments_merged_with_date.pkl')
df = pd.merge(df, df_2, on='comment_id')
df.drop(columns=['id', 'post_x'], inplace=True)
df.rename(columns={'post_y': 'post'}, inplace=True)
df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')

#%%
#df.to_pickle('all_comments_final.pkl')
df.to_csv('all_comments_final.csv.gz', index=False, compression='gzip')




#%%

df = df[df.post == '16ubhpq']
start = df['created'].min() 
eighteen_h = start + pd.Timedelta(18, 'h', hours=18)          
before = df[df['created'] < eighteen_h]
after = df[df['created'] > eighteen_h]


#%%
'''
instances = [
    "Earlier Monday evening, Pahlavi addressed a private audience and urged 'civil disobedience by means of non-violence.'",
    "I am a proponent of civil disobedience and logic driven protest only; not non/ irrational violence, pillage & mayhem!"
]
'''
#instances = before['text'].tolist()
instances = [text for text in before['text'] if isinstance(text, str)]


#maybe remove tags like YTA che magari è quello che considera come care

#model = models["care"]
for foundation in FOUNDATIONS:
    model = models[foundation]
    inputs = tokenizer(
        instances,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(model.device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate class probability
    probs = torch.softmax(outputs.logits, dim=1)
    probs = probs[:, 1]
    probs = probs.detach().cpu().numpy()

    before['probs_'+foundation] = probs


# %%
#


for subm_id, sub_df in df.groupby('post'):
    start = sub_df['created'].min() 
    eighteen_h = start + pd.Timedelta(18, 'h', hours=18)  
    before = sub_df[sub_df['created'] < eighteen_h]