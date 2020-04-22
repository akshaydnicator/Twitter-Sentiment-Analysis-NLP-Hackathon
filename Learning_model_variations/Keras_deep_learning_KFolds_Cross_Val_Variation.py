# Import required libraries

import numpy as np
import pandas as pd
import re
import pickle

# Load any variation of tweet embeddings created from Spacy, BERT and ELMo the state-of-the-art NLP models
# and assign it to X
pickle_in = open("Spacy_elmo_train.pickle","rb")     # Also try Spacy_bert_train.pickle or Spacy_bert_elmo_train.pickle or bert_elmo_train.pickle
X = pickle.load(pickle_in)
print(X.shape)

# Load train dataset
df = pd.read_csv('train.csv')
print(df.head())

# Check the distribution of target variable for binary classification with '0' being positive and '1' being negative tweet
print(df['label'].value_counts())


# Prepare the target variable for keras model
from keras.utils import to_categorical
y = df['label']
y = to_categorical(y)
print(y.shape)

# Another variation of splitting the dataset into train and test subsets, takes up less memory than sklearn_train_test_split function
def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)        # should be int
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    y_train = target[ratio:,:]
    y_test =  target[:ratio,:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = shuffle(X, y, 5)     # splits the dataset in the ratio of 1/5


## If using Conv1D layer, reshape the data from 2-dim to 3-dim to feed as an input
## If not using Conv1D, might just comment out this section and the Conv1D layers during model building
X_train = np.reshape(X_train, X_train.shape + (1,))
X_test = np.reshape(X_test, X_test.shape + (1,))

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)



### Build Keras Deep Learning model

# Import the layers that you want to use to build model and visualize key metrics
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten   #, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from sklearn.model_selection import KFold           # To be used for cross-validation
from matplotlib import pyplot


## Try different combinations of layers and their respective hyperparameters
## Comment out the layers which are not desired. Also, keep an eye on batch_input_shape parameter for Conv1D    

num_folds = 10                  # Select the number of folds for cross-validation
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)


# K-fold Cross Validation model building

fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    model = Sequential()
    model.add(Conv1D(64, 8, activation='relu', use_bias=True, bias_initializer='zeros', batch_input_shape=(None,1324,1)))
    #model.add(BatchNormalization(momentum=0.99, epsilon=0.0005))
    model.add(Flatten())    
    model.add(Dense(64, activation='relu'))                 #input_dim=(1324,)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.25, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    #model.summary()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35)
    relr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001, mode='min', verbose=1, min_delta=1E-7)
    
    # Save best model per fold to be used later on for comparison, cross validation and predictions
    ch = ModelCheckpoint(f'bestmodel_{fold_no}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=25)
    
    callbacks_list = [es,ch,relr]


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train],targets[train],epochs=1500, verbose=2, batch_size=8192, validation_split=0.3, callbacks=callbacks_list)
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=1)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    print(f'Saving final model per fol... {fold_no}')
    model.save(f'Fold_{fold_no}.hdf5')
    
    #Plotting charts
    print('\nPerformance Charts\n')
    pyplot.plot(history.history['acc'], label='training_accuracy')
    pyplot.plot(history.history['val_acc'], label='validation_accuracy')
    pyplot.plot(history.history['loss'], label='training_loss')    
    pyplot.plot(history.history['val_loss'], label='validation_loss')
    pyplot.show() 
        
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
print('Saving final model finally !!!.....')
model.save("Final.hdf5")

for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')



# Evaluate the model using loss and accuracy metrics
model.evaluate(x=X_test,y=y_test)

# Make predictions on X_test using latest updated trained model
predictions = model.predict_classes(X_test)
print(predictions)

print(y_test.argmax(axis=1))

# Import metrics from sklearn
from sklearn import metrics

# Print a confusion matrix
print(metrics.confusion_matrix(y_test.argmax(axis=1),predictions))

# Print a classification report
print(metrics.classification_report(y_test.argmax(axis=1),predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test.argmax(axis=1),predictions))

# Load test dataset for making final predictions using the latest updated instance of trained model
df1 = pd.read_csv('test.csv')
print(df1.head())

# Load corresponding tweet embeddings for test dataset same as it was loaded for training dataset
pickle_in = open("Spacy_elmo_test.pickle","rb")
test_X = pickle.load(pickle_in)
print(test_X.shape)

# You may have to reshape the test tweet embeddings dataset in order to feed into Conv1D layer of the model
# However comment out this step if not using Convolution Neural Network layer in the model
test_X = np.reshape(test_X, test_X.shape + (1,))
print(test_X.shape)

# Make prediction on the test dataset for final submission
test_predictions = model.predict_classes(test_X)
print(test_predictions)
df1['label'] = test_predictions
df1 = df1.drop(['tweet'],axis=1)
print(df1.head())

# Saving the final predictions to csv file
df1.to_csv('spacy_elmo_final.csv', index=False)


# Now loading the best model that got saved using ModelCheckpoint Callback and its hyperparameters
from keras.models import load_model
model1 = load_model('bestmodel_3.hdf5')

# Making predictions on test dataset using the best saved model
best_predictions = model1.predict_classes(test_X)
print(best_predictions)

# Making a copy of dataframe to store newly predicted sentiment values
df2 = df1
df2['label'] = best_predictions
print(df2.head())

# Saving the best predicted version of test dataset
df2.to_csv('spacy_elmo_best_final.csv')