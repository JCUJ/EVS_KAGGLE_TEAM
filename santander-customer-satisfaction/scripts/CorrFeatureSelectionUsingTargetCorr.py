# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:44:25 2020

@author: Alejandro
"""

#Correlation

import os
import time
import pandas as pd
import numpy as np

#Correlation function

# Predictors and response variable
X = pd.DataFrame(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), columns=['a','b','c','d'])
y = pd.DataFrame(np.array([100,1000,0,2]))

# 1.- Estimate correlation against repsonse variable
# We sould be using this: targetCorr = X.corrwith(y) but let's just define it.
targetCorr = pd.DataFrame(np.array([.8,.1,.2,.7]), index = X.columns.values)
# 2.- Sort correlated values
targetCorr.sort_values(0, ascending=False, inplace = True)
# 3.- Sort X
X = X[targetCorr.index] # New sorted X set
# 4.- Find X correlation matrix. For practice purposes we'll define it.
corrMatrix = pd.DataFrame(np.array([[1,.7,.2,.1],[.7,1,.2,.7],[.2,.2,1,.7],[.1,.7,.7,1]]), columns=['a','d','c','b'])

# 5.- Correlation function
def corr_feature_sel(corr_df, threshold):
    columns = np.full(corr_df.shape[0], True, dtype=bool)

    for i in range(corr_df.shape[0]):
        for j in range(i+1,corr_df.shape[0]):
            if corr_df.iloc[i,j] >= threshold:
                if columns[j]:
                    columns[j]=False
    return columns
                    
# 6.- Apply function to corrMatrix and choose a threshold
columns = corr_feature_sel(corrMatrix,.6)
selColumns = X.columns[columns] # Selected features
newX = X[selColumns] # X subset with selected features.

