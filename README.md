# Twitter Sentiment Analysis NLP Hackathon by AV
## Problem Statement
Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. This time around, given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, the task is to identify if the tweets have a negative sentiment towards such companies or products.

## Data Description
For training the models, a labelled tweets dataset is provided. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet. The test data file contains only tweet ids and the tweet text with each tweet in a new line.
### Dataset
- train.csv: 7,920 tweets
- test.csv: 1,953 tweets

## Approach and Implementation
### Data Preparation
Both the training and test file tweets have been **pre-processed using one standardized process** as given below:
- Removing any URLs, twitter user handles, punctuation marks, numbers and whitespaces from the train and test tweets along with converting them to lower case and lemmetizing further to normalize the text datasets to feed into different vector models
- Using **Spacy, BERT and ELMo models** to create tweet embeddings with dimensions as (300,), (768,) and (1024,) respectively
- Using various combinations of the three state-of-the-art vector models so as to create vector arrays of dimenstions (300+768,), (300+1024,), (768+1024,) and (300+768+1024,). These feature vectors were used as an input to feed into different classification models
- Apart from that, two other features were also extracted from tweets. Using **data exploration techniques** it was realized that the usage of **length of tweet and length of punctuation marks** in a tweets could increase the accuracy of a standard LinearSVC model when used in conjuntion with Tf-idf vectorizer. Hence, these features were also normalized using sklearn **MinMaxScaler** and added as additional features to earlier feature vectors to feed into different classification models
### Models Used and Evaluation
   From the **sklearn machine learning library**, two variants of vectorizers namely **CountVectorizer and Tf-idf Vectorizers** were used apart from the vectors obtained earlier from the combinations of **Spacy, BERT, ELMo and text features.** These were then fed into five different variants of classifiers that were used for binary classification problem, whereby label '0' depicted a positive opinion and '1' depicted a negative opinion in a tweet. Further, different aspects of **KFold** library were used as a combination for making different sets of train and test data for **cross validation** of the learning models.
   - A variant of feature vectors and **LinearSVC** achieved an f1 score of 88.40 on the test dataset. f1 score is the standard score Analytics Vidhya uses to evaluate submissions
   - A variant of feature vectors and **LogisticRegression** achieved an f1 score of 87.56 on the test dataset
   - A variant of feature vectors and **RandomForestClassifier** achieved an f1 score of 81.96 on the test dataset
   - A variant of feature vectors and **DecisionTreeClassifier** achieved an f1 score of 77.36 on the test dataset
   - A variant of feature vectors and **KNeighborsClassifier** achieved an f1 score of 86.44 on the test dataset

   Further, **Keras** a high-level neural networks API was used with **TensorFlow** as backend to build variations of **Multilayer Perceptron (MLP) models.** A variety of feature vectors and a variety of MLP models with a combination of different layers including **Conv1D, Dense, Flatten, BatchNormalization and Droupout,** and with variations of optimizers including **Adam and SGD** along with further tuning of their **respective hyperparameters** inluding but not limited to number of nodes, number of MLP layers, relative position of layers, learning rate, batch size, validation split, momemtum, loss, decay and a variation of activation funtions including **relu, LeakyReLU, softmax and sigmoid** with control measures using important Callbacks including **EarlyStopping, ModelCheckpoint, ReduceLROnPlateau** were used to build a variety of different deep learning models for the training dataset. 

## Result
A variation of different combinations of models led to the **highest f1 score of 91.44** on the test dataset and it **ranked #10 on public leaderboard** as on April 21, 2020 with 4733 total number of participants having made 686 number of unique submissions.  
## Further Scope of Study
- The results may further be refined using other combinations of word embeddings such as GloVe, Word2Vec and fastText
- Ensemble of different learning algorythms might be used to increase the prediction accuracy of models further

## Acknowlegdement
Many snippets of the code used may have been taken from other open GitHub repositories to ease the rapid production and pace up the flow in the competition. It is acknowledged here that data has been gathered from multiple sources. I am thankful to all of them for their mentorship and help.
