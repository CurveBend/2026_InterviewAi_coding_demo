####### BERT sentiment analysis for interview data #####
# script: Han Olff, 30 apr 2026
# for documentation see: https://huggingface.co/docs/transformers/en/main_classes/pipelines

# best to run Positron for this script as an administrator
# on first time use of this script create the virtual environment .venv folder 
# in the terminal, run: 
# python -m venv .venv

# make sure interpreter Python 3.14.4 in Venv is selected, not R (topright in the screen under variables)

# during the first time working with this script, run the following commands in the terminal below
# to install the required libraries
# pip install pandas
# pip install transformers
# pip install torch

# in this script we use the model distilbert-base-uncased-finetuned-sst-2-english
# this model  is described here (including warnings)
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
# this model comes from Hugging Face (huggingface.co),  an online platform for sharing,
# using, and building machine learning models—especially for natural language processing and, 
# increasingly, vision and audio.
# think of it as github for AI models and datasets

# load the required libraries
import pandas as pd
from transformers import pipeline

# Force pipeline to use CPU, avoids GPU .numpy() issues
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)
# warning is given, how to prevent
# Create an account on Hugging Face
# Generate a token: https://huggingface.co/settings/tokens
# In your terminal:
# huggingface-cli login
# or set it as environment variable in your terminal:
# setx HF_TOKEN "your_token_here"

# Load the example  dataset of the text responses to Question 1 of the survey
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRjv0h7adkomA-Z0oNSNVtZMtXdzdAoJI-RRSIGJBTFDWDkUnuVQ7YIp17o7DuZ0ShAJzsEFa5EyIku/pub?gid=1895111920&single=true&output=csv"
df_all = pd.read_csv(url)
# filter for only answers to he first question
df = df_all[df_all["Question_ID"] == "Q01_ClimateImpact"]

# show the dataframe
df

# Load sentiment pipeline (positive/negative)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load emotion classification pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    top_k=None
)

# Apply sentiment analysis
df['sentiment'] = df['Response_txt'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
df['sentiment_score'] = df['Response_txt'].apply(lambda x: sentiment_pipeline(x)[0]['score'])

# Apply emotion classification and extract top emotion
def get_top_emotion(text):
    scores = emotion_pipeline(text)[0]
    top = max(scores, key=lambda x: x['score'])
    return top['label'], top['score']

df[['top_emotion', 'emotion_score']] = df['Response_txt'].apply(
    lambda x: pd.Series(get_top_emotion(x))
)

# Show result
print(df[['Response_ID', 'sentiment', 'sentiment_score', 'top_emotion', 'emotion_score']])
