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

# load data 
from keras.datasets import mnist
import keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# preprocess data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# creating the basic model 
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_callback = model.fit(X_train, y_train, batch_size=128,
                             epochs=20, verbose=1, validation_split=.1)
                          
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

# plot the learning curve
ef plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")
    return plt

plot = plot_history(history_callback)
plot.savefig("lc_1.png")

## basic model adding drop out
from keras.layers import Dropout

model_dropout = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dropout(.5),
    #Dense(1024, activation='relu'),
    #Dropout(.5),
    Dense(10, activation='softmax'),
])
model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_dropout = model_dropout.fit(X_train, y_train, batch_size=128,
                            epochs=20, verbose=1, validation_split=.1)

## result
score = model_dropout.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

## plot learning curve
plot = plot_history(history_dropout)
plot.savefig("lc_1_drop_basic.png")

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

# Function to create model, required for KerasClassifier
def create_model(neurons=32,dropout_rate=0.2, weight_constraint=4):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_shape=(784,),  activation='relu', W_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10,  activation='softmax'))
    # Compile model
    model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
    return model

# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=20, batch_size=128, verbose=1)

# define the grid search parameters
neurons = [32, 64, 256, 1024]
# weight_constraint = [0, 2, 4]
dropout_rate = [ 0.2,  0.5,  0.8]
param_grid = dict(neurons=neurons, dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1)

grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# get the best params
neurons = grid_result.best_params_['neurons']
dropout_rate = grid_result.best_params_['dropout_rate']


# creating the basic model using best params
model = Sequential([
    Dense(neurons, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_callback = model.fit(X_train, y_train, batch_size=128,
                             epochs=20, verbose=1, validation_split=.1)
                          
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

plot = plot_history(history_callback)
plot.savefig("lc.png")

##  model with drop out and best params
from keras.layers import Dropout

model_dropout = Sequential([
    Dense(neurons, input_shape=(784,), activation='relu'),
    Dropout(.5),
    #Dense(1024, activation='relu'),
    #Dropout(dropout_rate),
    Dense(10, activation='softmax'),
])
model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_dropout = model_dropout.fit(X_train, y_train, batch_size=128,
                            epochs=20, verbose=1, validation_split=.1)

## result
score = model_dropout.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

## plot learning curve
plot = plot_history(history_dropout)
plot.savefig("lc_drop_basic.png")
