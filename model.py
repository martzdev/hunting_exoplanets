#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mihaisturza
"""

# importing libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# create X, y for selected dataset
def create_dataset(filename):
    dataset = pd.read_csv(filename)
    # time steps
    X = dataset.iloc[:,1:].values
    # labels
    y = dataset.iloc[:,0:1].values
    return X, y

# create model
def create_model(shape):
    # We are creating a CNN
    model = Sequential()
    
    # Size reduction
    # Conv layer
    model.add(Conv1D(64,kernel_size=10,input_shape=shape,activation="relu"))
    # Pooling
    model.add(MaxPooling1D(strides=4))
    model.add(BatchNormalization())
    # Some dropout
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # FC
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation="softmax"))

    # Complie
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    
    return model

# Scale our data
sc = StandardScaler()
def scale_data(data,first_run):
    # Check if our scaler has been already fitted on a dataset
    # And return it reshaped in a 3D Tensor for our CNN
    if first_run:
        return np.reshape(sc.fit_transform(data),(data.shape[0],data.shape[1],1))
    else:
        return np.reshape(sc.transform(data),(data.shape[0],data.shape[1],1))

# Train our model
def train_model(model, X_train, y_train, batch, epochs):
    # Create a callback that will save our model after each epoch
    callback = ModelCheckpoint("checkpoint.h5",save_weights_only=True)
    model.fit(X_train, y_train, batch_size=batch, epochs=epochs, callbacks=[callback])
    return model

# Evaluate our model
def eval_model(model, X_test, y_test):
    # TF evaluate functionb
    evaluate = model.evaluate(X_test,y_test)
    print(f"Accuracy: {evaluate[1]}\nLoss: {evaluate[0]}")
    # Confusion matrix
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: {conf_matrix}\nCorrect Values: {conf_matrix[0][0]+conf_matrix[1][1]}\nIncorrect Values: {conf_matrix[0][1]+conf_matrix[1][0]}")

def main():
    # Create and split our data
    X_train, y_train = create_dataset("dataset/exoTrain.csv")
    X_test, y_test = create_dataset("dataset/exoTest.csv")
    # Scale the date
    X_train = scale_data(X_train,1)
    X_test = scale_data(X_test,0)
    # Create a model with the shape of our data
    model = create_model(X_train.shape[1:])
    # Train the model
    model = train_model(model, X_train,y_train, 32, 4)
    # Finally evaluate it!
    eval_model(model, X_test, y_test)

# Run everything
if __name__ == "__main__":
    main()