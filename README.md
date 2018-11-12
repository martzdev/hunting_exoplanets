# Hunting Exoplanets
Using ML to find exoplanets - Dataset from Kaggle

## Understanding our data
Source - [Exoplanet Hunting in Deep Space](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data)
### 1. Take care of labels
  - 2 is an exoplanet star and 1 is a non-exoplanet-star according to the dataset description
  - Change that to 1 for exoplanets and 0 for non exoplanets
### 2. Understand how exoplanets are labeled
  - To do this we create plots for exoplanets and non exoplanets luminosity
  - After doing so we can see that exoplanets have a greater luminosity than non exoplanets
  ![](https://raw.githubusercontent.com/martzdev/hunting_exoplanets/master/images/exo_lum.png)
  ![](https://raw.githubusercontent.com/martzdev/hunting_exoplanets/master/images/nonexo_lum.png)
  - We are also comparing the mean of the luminosity so we can have a better understanding
  ![](https://raw.githubusercontent.com/martzdev/hunting_exoplanets/master/images/exo_mean.png)
  ![](https://raw.githubusercontent.com/martzdev/hunting_exoplanets/master/images/nonexo_mean.png)
  
