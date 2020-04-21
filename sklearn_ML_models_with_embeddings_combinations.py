# Import required libraries

import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_colwidth', 100)

# Load any variation of word embeddings from Spacy, BERT and ELMo and assign it to X variable
pickle_in = open("Spacy_bert_elmo_train.pickle","rb")      # Also try Spacy_bert_train.pickle or Spacy_elmo_train.pickle or bert_elmo_train.pickle
X = pickle.load(pickle_in)
X.shape

# Load the training dataset into a dataframe
df = pd.read_csv('train.csv')
print(df.head())

# Observe the distribution of positive and negative tweets
print(df['label'].value_counts())

# Assign targets to y variable
y = df['label']

# Split the training dataset into train and test subsets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


### Load any of the five models given below; LinearSVC, LogisticsRegression, KneighborsClassifier, RandomForestClassifier or DecisionTreeClassifier
## If using raw text, may use a variant of Vectorizer given below as CountVectorizer and TfidfVectorizer 
## placed in front of an ML model in the pipeline 

from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import MaxAbsScaler

#knn = KNeighborsClassifier(n_neighbors=8)      # Initializing KNNClassifier


## Create a pipeline so as to streamline the flow of the data into the ML model
text_clf = Pipeline([('clf', LinearSVC())]) #('tfidf', TfidfVectorizer()), ('scaler', MaxAbsScaler())

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)

# Make predictions
predictions = text_clf.predict(X_test)
print(predictions)


# Import metrics to check the accuracy and other features of the predictions made
from sklearn import metrics

# Print a confusion matrix
print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))

# Loading test dataset
df1 = pd.read_csv('test.csv')
print(df1.head())

# Loading corresponding test tweets embeddings as loaded for the training dataset
pickle_in = open("spacy_bert_elmo_test.pickle","rb")
test_X = pickle.load(pickle_in)
print(test_X.shape)

# Dropping tweet column as it is no longer required for final submission leaving only the id column for identification
df1 = df1.drop(['tweet'],axis=1)
print(df1.head())

# Making predictions using trained model for the test dataset for final submission 
test_predictions = text_clf.predict(test_X)
print(test_predictions)

# Adding predicted labels to the test dataframe
df1['label'] = test_predictions
print(df1.head())

# Saving the final predicted submission file to csv
df1.to_csv('ALL_SVM.csv', index=False)

