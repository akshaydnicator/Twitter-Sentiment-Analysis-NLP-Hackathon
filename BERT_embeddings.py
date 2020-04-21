# The original notebook is hosted on Kaggle. Link -> https://www.kaggle.com/akdnic/bert-tweets-to-vectors
# This copy of code is for references only

# Import required libraries
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle

# Install bert-as-service
!pip install bert-serving-server
!pip install bert-serving-client

# Download and unzip the pre-trained model
!wget http://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip

# Start the BERT server
bert_command = 'bert-serving-start -model_dir /kaggle/working/uncased_L-12_H-768_A-12'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)

# Start the BERT client
from bert_serving.client import BertClient
bc = BertClient(check_length=False)

# Load the training dataset
df = pd.read_csv('../input/hackathon-train/train.csv')
print(df.head())

# Create a list of punctuation marks
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

# Code to replace punctuations with whitespaces
def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' ')
    return x

# Cleaning URLs, twitter user_handles, punctuations, whitespaces and converting to lowercase
df['Cleaned'] = df['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
df['Cleaned'] = df['Cleaned'].apply(lambda x: re.sub("@[\w]*", '', x))
df['Cleaned'] = df['Cleaned'].apply(lambda x: clean_text(x))
df['Cleaned'] = df['Cleaned'].str.lower()
df['Cleaned'] = df['Cleaned'].apply(lambda x:' '.join(x.split()))
df['Sentiment'] = df['label']
df = df.drop(['tweet','label'],axis=1)
print(df.head())

# Compute embeddings for training tweets using Bert Client encode function
# The model returns 768-dimensional embeddings
tweets = df['Cleaned']
tweet_list = [word for word in tweets]
embeddings = bc.encode(tweet_list)

print('embeddings.shape')

# save bert_train_new for reuse as it would take a really long time for conversion
pickle_out = open("bert_train.pickle","wb")
pickle.dump(embeddings, pickle_out)
pickle_out.close()

# Loading test dataset
df1 = pd.read_csv('../input/hackathon-test/test.csv')
df1.head()

# Cleaning the test dataset similarly as train dataset
df1['Cleaned'] = df1['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
df1['Cleaned'] = df1['Cleaned'].apply(lambda x: re.sub("@[\w]*", '', x))
df1['Cleaned'] = df1['Cleaned'].apply(lambda x: clean_text(x))
df1['Cleaned'] = df1['Cleaned'].str.lower()
df1['Cleaned'] = df1['Cleaned'].apply(lambda x:' '.join(x.split()))
df1 = df1.drop(['tweet'],axis=1)
df1.head()

# Compute Bert embeddings for some test tweets
test_tweets = df1['Cleaned']
test_tweet_list = [words for words in test_tweets]
test_embeddings = bc.encode(test_tweet_list)
test_embeddings.shape

# save bert_test_new
pickle_out = open("bert_test.pickle","wb")
pickle.dump(test_embeddings, pickle_out)
pickle_out.close()