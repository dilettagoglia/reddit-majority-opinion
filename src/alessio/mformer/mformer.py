#%%
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import os
import torch
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


df = pd.read_pickle('/Users/alessiogandelli/dev/uni/reddit-disagreement/data/data-tidy/all_comments_merged.pkl')


#%%
instances = [
    "Earlier Monday evening, Pahlavi addressed a private audience and urged 'civil disobedience by means of non-violence.'",
    "I am a proponent of civil disobedience and logic driven protest only; not non/ irrational violence, pillage & mayhem!"
]

instances = df['text'].tolist()
instances = [text for text in df['text'] if isinstance(text, str)]


#maybe remove tags like YTA che magari è quello che considera come care



model = models["care"]

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

    df['probs_'+foundation] = probs


# %%
