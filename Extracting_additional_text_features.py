# Import required libraries

import numpy as np
import pandas as pd
import re

# Load training dataset
df = pd.read_csv('train.csv')
print(df.head())

# Function to calculate length of characters in a tweet
def tweet_length(x):
    length = len(x)
    return length

# Calculate and append tweet length to dataframe
df['tweet_len'] = df['tweet'].apply(lambda x: tweet_length(x))


# Create a list of punctuations
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

# Function to calculate the total length of punctuation marks in a tweet
def puncts_len(x):
    punct_list = []
    x = str(x)
    for punct in puncts:
        for char in x:
            if punct==char:
                punct_list.append(punct)
    return len(punct_list)

# Calculate length of punctuation marks
df['punct_len'] = df['tweet'].apply(lambda x: puncts_len(x))
print(df.head())

# Describing the basic statistics including mean, count, std etc for extracted tweet lengths and punctuation marks length
print(df['tweet_len'].describe())
print(df['punct_len'].describe())

# Using matlotlib to visualize the distribution of length of tweets
import matplotlib.pyplot as plt
#%matplotlib inline

bins = 1.15**(np.arange(0,50))
plt.hist(df['tweet_len'],bins=bins,alpha=0.8)
plt.hist(df[df['label']==1]['tweet_len'],bins=bins,alpha=0.8)
plt.legend(('pos','neg'))
plt.show()

# Visualizing the distribution of length of punctuation marks
bins = 1.15**(np.arange(0,30))
plt.hist(df['punct_len'],bins=bins,alpha=0.8)
plt.hist(df[df['label']==1]['punct_len'],bins=bins,alpha=0.8)
plt.legend(('pos','neg'))
plt.show()

# Saving additional features with training dataset for further use
df.to_csv('train_more_features.csv')


# Similarly for test dataset loading new dataframe
df1 = pd.read_csv('test.csv')
print(df1.head())

# Calculating tweet length and punctuation marks length in test dataset and adding to dataframe
df1['tweet_len'] = df1['tweet'].apply(lambda x: tweet_length(x))
df1['punct_len'] = df1['tweet'].apply(lambda x: puncts_len(x))
print(df1.head())

# Saving additional features with test dataset for further use
df1.to_csv('test_more_features.csv')
