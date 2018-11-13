#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mihaisturza
"""
# Here we will take a short look over our data so we can understand it.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# First, let's prepare the dataset
def create_dataset():
    # Read it
    dataset = pd.read_csv("dataset/exoTrain.csv") # Dataset from Kaggle - view README.md
    # Replace labels
    # In the original format
    #   2 = Exoplanet
    #   1 = Not exoplanet
    # We replace 1 with 0 and 2 with 1
    dataset['LABEL'] = dataset['LABEL'].replace([1],[0])
    dataset['LABEL'] = dataset['LABEL'].replace([2],[1])

    # return everything as a numpy array
    return np.array(dataset)

# We should understand how exoplanets get detected
# To do this, we're using the only feature in our dataset: luminosity
def plot_luminosity(data,label):
    # Simply plot all the luminosity detected in the time steps
    plt.plot(data)
    # Set the correct title
    if label == 1:
        plt.title("Exoplanet")
    else:
        plt.title("Not Exoplanet")
    # Prepare the axis
    plt.xlabel("Time Series")
    plt.ylabel("Luminosity")
    # Plot!
    plt.show()
    
# Let's see how an exoplanet's ,mean luminosity looks like compared to a non exoplanet
def plot_mean(data,exo):
    # Let's see what we want
    # Explanet
    if exo == 1:
        # Create a list that stores every mean
        mean_exo = []
        # Iterate throughout the dataset
        for row in data:
            if row[0]==1:
                # Finally, append the mean
                mean_exo.append(row[1:].mean())
        plt.hist(mean_exo)
        plt.title("Exoplanet")
    # Not Exoplanet
    else:
        # Do the same as above
        mean_notexo = []
        for row in data:
            if row[0]==0:
                mean_notexo.append(row[1:].mean())
        plt.hist(mean_notexo)
        plt.title("Not Exoplanet")
    # Prepare the axis
    plt.xlabel('Mean intensity')
    plt.ylabel('Stars')
    # Plot!
    plt.show()

# Now, let's see everything in action
def main():
    # Create dataset
    data = create_dataset()
    # Plot the graphs for exoplanets
    plot_luminosity(data[0,1:],1)
    plot_mean(data,1)
    # Plot the graphs for non exoplanets
    plot_luminosity(data[100,1:],0)
    plot_mean(data,0)

if __name__ == "__main__":
    # Run!
    main()
