# Twitter Sentiment Analysis Competition by Analytics Vidhya
## Problem Statement: 
Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. This time around, given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, the task is to identify if the tweets have a negative sentiment towards such companies or products.

## Data Description
For training the models, a labelled dataset tweets is provided. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet. The test data file contains only tweet ids and the tweet text with each tweet in a new line.
### Dataset
- train.csv: 7,920 tweets
- test.csv: 1,953 tweets

## Approach and Implementation
### Data Preparation
Both the training and test file tweets have been pre-processed using one standardized process as given below:
- Removing any URLs, twitter user handles, punctuation marks, numbers and whitespaces from the train and test tweets along with converting them to lower case and lemmetaizing further to normalize the text datasets to feed into different vector models
- Using **Spacy, BERT and ELMo models** to create tweet embeddings with dimensions as (300,), (768,) and (1024,) respectively
- Using various combinations of the three state-of-the-art vector models so as to create vector arrays of dimenstions (300+768,), (300+1024,), (768+1024,) and (300+768+1024,). These feature vectors were used as an input to feed into different classification models
- Apart from that, two other features were also extracted from tweets. Using **data exploration techniques** it was realized that the usage of length of tweets and length of punctuation marks could increase the accuracy of a standard LinearSVC() model when used in conjuntion with Tf-idf vectorizer. Hence, these features were also normalized using sklearn MinMaxScaler() and added as additional features to earlier feature vectors to feed into different classification models
### Models Used and Evaluation
1. From the sklearn machine learning library, five different variants of classifiers were used for binary classification problem, whereby label '0' depicted a positive opinion and '1' depicted a negative opinion in a tweet
