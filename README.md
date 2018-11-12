# Hunting Exoplanets
Using ML to find exoplanets - Dataset from Kaggle

## Understanding our data
Source - [Exoplanet Hunting in Deep Space](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data)
### 1. Take care of labels
  - 2 is an exoplanet star and 1 is a non-exoplanet-star according to the dataset description
  - Change that to 1 for exoplanets and 0 for non exoplanets
### 2. Understand how exoplanets are labeled
  - To do this we create plots for exoplanets and non exoplanets
  - After doing so we can see that exoplanets have a greater luminosity than non exoplanets
