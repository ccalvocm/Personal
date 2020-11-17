# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:19:30 2020

@author: ccalvo
"""

#%% Con Ordinary Least Squares
import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn
import numpy as np
import pandas as pd

data = datasets.load_boston() ## loads Boston dataset from datasets library 

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])


## Without a constant

X = df["RM"]
y = target["MEDV"]
X = sm.add_constant(X) 

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()

#%% Con SKLearn

from sklearn import linear_model
from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target['MEDV']

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)

lm.score(X,y)

lm.coef_
lm.intercept_



