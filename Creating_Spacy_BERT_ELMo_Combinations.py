# Import required libraries
import pandas as pd
import numpy as np
import pickle

pd.set_option('display.max_colwidth', 100)

# Load Spacy_train Vectors
pickle_in = open("Spacy_train.pickle","rb")
spacy_train = pickle.load(pickle_in)

# Load Spacy_test Vectors
pickle_in = open("Spacy_test.pickle","rb")
spacy_test = pickle.load(pickle_in)

print(spacy_train.shape, spacy_test.shape)

# Load BERT_train Vectors
pickle_in = open("bert_train.pickle","rb")
bert_train = pickle.load(pickle_in)

# Load BERT_test Vectors
pickle_in = open("bert_test.pickle","rb")
bert_test = pickle.load(pickle_in)

print(bert_train.shape, bert_test.shape)

# Load ELMo_train Vectors
pickle_in = open("elmo_train.pickle","rb")
elmo_train = pickle.load(pickle_in)

# Load ELMo_test Vectors
pickle_in = open("elmo_test.pickle","rb")
elmo_test = pickle.load(pickle_in)

print(elmo_train.shape, elmo_test.shape)


# Create Spacy + BERT Vectors
spacy_bert_train = np.hstack((spacy_train, bert_train))
spacy_bert_test = np.hstack((spacy_test, bert_test))

print(spacy_bert_train.shape, spacy_bert_test.shape)

# save spacy_bert_train
pickle_out = open("Spacy_bert_train.pickle","wb")
pickle.dump(spacy_bert_train, pickle_out)
pickle_out.close()

# save Spacy_bert_test
pickle_out = open("Spacy_bert_test.pickle","wb")
pickle.dump(spacy_bert_test, pickle_out)
pickle_out.close()

# Create BERT + ELMo Vectors
bert_elmo_train = np.hstack((bert_train, elmo_train))
bert_elmo_test = np.hstack((bert_test, elmo_test))

print(bert_elmo_train.shape, bert_elmo_test.shape)

# save bert_elmo_train
pickle_out = open("bert_elmo_train.pickle","wb")
pickle.dump(bert_elmo_train, pickle_out)
pickle_out.close()

# save bert_elmo_test
pickle_out = open("bert_elmo_test.pickle","wb")
pickle.dump(bert_elmo_test, pickle_out)
pickle_out.close()

# Create Spacy + ELMo Vectors
spacy_elmo_train = np.hstack((spacy_train, elmo_train))
spacy_elmo_test = np.hstack((spacy_test, elmo_test))

print(spacy_elmo_train.shape, spacy_elmo_test.shape)

# save Spacy_elmo_train
pickle_out = open("Spacy_elmo_train.pickle","wb")
pickle.dump(spacy_elmo_train, pickle_out)
pickle_out.close()

# save Spacy_elmo_test
pickle_out = open("Spacy_elmo_test.pickle","wb")
pickle.dump(spacy_elmo_test, pickle_out)
pickle_out.close()

# Create Spacy + BERT + ELMo Vectors
spacy_bert_elmo_train = np.hstack((spacy_train, bert_train, elmo_train))
spacy_bert_elmo_test = np.hstack((spacy_test, bert_test, elmo_test))

print(spacy_bert_elmo_train.shape, spacy_bert_elmo_test.shape)

# save bert_elmo_train
pickle_out = open("Spacy_bert_elmo_train.pickle","wb")
pickle.dump(spacy_bert_elmo_train, pickle_out)
pickle_out.close()

# save bert_elmo_test
pickle_out = open("Spacy_bert_elmo_test.pickle","wb")
pickle.dump(spacy_bert_elmo_test, pickle_out)
pickle_out.close()