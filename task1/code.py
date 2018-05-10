
# import packages
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
plt.rcParams["figure.dpi"] = 200
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation

# preprocess data set
from sklearn import datasets
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns= iris['feature_names']+['target'])
X = data.iloc[:,0:4].astype(float)
Y = data.iloc[:,4].astype(int)

# Convert labels to categorical one-hot encoding
import keras
binary_Y = keras.utils.to_categorical(Y, num_classes=3)

# split data into train and test sets
X = np.asarray(X)
binary_Y = np.asarray(binary_Y)
X_train, X_test, y_train, y_test = train_test_split(X, binary_Y, stratify=binary_Y, random_state = 0)

# create basic keras model
model = Sequential()
model.add(Dense(4, activation='relu', input_dim=4))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
# fit the basic model and evaluate the model using test sets
model.fit(X_train, y_train,batch_size=5, epochs=20, verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

# model selection by grid search

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

def make_model(optimizer="adam", hidden_size_1=4, hidden_size_2=4):
    model = Sequential()
    model.add(Dense(hidden_size_1, activation='relu', input_dim=4))
    model.add(Dense(hidden_size_2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

param_grid = {'epochs': [10],  
              'hidden_size_1': np.arange(4,20,5),# epochs is fit parameter, not in make_model!
              'hidden_size_2': np.arange(4,20,5)}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# get the best parameters and create the best model
hidden_size_1 = grid.best_params_['hidden_size_1']
hidden_size_2 = grid.best_params_['hidden_size_2']
model_best = Sequential()
model_best.add(Dense(hidden_size_1, activation='relu', input_dim=4))
model_best.add(Dense(hidden_size_2, activation='relu'))
model_best.add(Dense(3, activation='softmax'))
model_best.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_best.fit(X_train, y_train,batch_size=5, epochs=20, verbose=1)
# get the best result
score_best = model_best.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score_best[0]))
print("Test Accuracy: {:.3f}".format(score_best[1]))
