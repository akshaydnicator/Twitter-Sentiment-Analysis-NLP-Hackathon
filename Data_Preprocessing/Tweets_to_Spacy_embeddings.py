# Import required libraries

import numpy as np
import pandas as pd
import re
import spacy
import pickle

pd.set_option('display.max_colwidth', 100)

# Load the largest english language vector collection from Spacy
nlp = spacy.load('en_vectors_web_lg')

# Read train and test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape, test.shape)

# Check the distribution of positive and negative target labels in the training dataset
print(train['label'].value_counts(normalize = True))

# Define a list of punctuation marks
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

# Code to replace punctuation marks with whitespace 
def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' ')
    return x

# Remove URL's from train and test datasets
train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

# Remove user handles @
train['clean_tweet'] = train['clean_tweet'].apply(lambda x: re.sub("@[\w]*", '', x))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: re.sub("@[\w]*", '', x))

# Remove punctuation marks
train['clean_tweet'] = train['clean_tweet'].apply(lambda x: clean_text(x))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: clean_text(x))

# Convert text to lowercase
train['clean_tweet'] = train['clean_tweet'].str.lower()
test['clean_tweet'] = test['clean_tweet'].str.lower()

# Remove numbers
train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")
test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

# Remove whitespaces
train['clean_tweet'] = train['clean_tweet'].apply(lambda x:' '.join(x.split()))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))

# Function to lemmatize the tokens to their basic forms to normalize the tweet text
# and focus on key words for the classification tasks

def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

# Record the time to lemmatize the tweets and converting them into vectors using spaCy
start_time = time.monotonic()

# Lemmatize the tokens
train['clean_tweet'] = lemmatization(train['clean_tweet'])
test['clean_tweet'] = lemmatization(test['clean_tweet'])

print(train.tail(), test.tail())
   

# Convert cleaned tweets into Spacy word vectors
# The model returns 300-dimensional embeddings
tweets = train['clean_tweet']
word_vec = [nlp(word).vector for word in tweets]
X_tr = np.array(word_vec)

test_tweets = test['clean_tweet']
test_word_vec = [nlp(word).vector for word in test_tweets]
X_te = np.array(test_word_vec)

print(X_tr.shape, X_te.shape)

end_time = time.monotonic()

# Print the time taken to finish the process by spaCy
print(f'Time taken to lemmitize and vectorize 1.6m tweets: {timedelta(seconds=end_time - start_time)}')

# Save Spacy_train_new
pickle_out = open("Spacy_train.pickle","wb")
pickle.dump(X_tr, pickle_out)
pickle_out.close()

# Save Spacy_test_new
pickle_out = open("Spacy_test.pickle","wb")
pickle.dump(X_te, pickle_out)
pickle_out.close()
