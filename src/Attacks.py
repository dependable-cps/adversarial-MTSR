# Regression models: How many more cycles an in-service engine will last before it fails?
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

class attacks():

# fast gradient sign method
    def fgsm(X, Y, model,epsilon,targeted= False):
        dlt = model.predict(X)- Y
        if targeted:
            dlt = model.predict(X) - Y
        else:
            dlt = -(model.predict(X) - Y)
        dir=np.sign(np.matmul(dlt, model.weight.T))
        return X + epsilon * dir, Y


#basic iterative method
    def bim(X, Y, model, epsilon, alpha, I):
        Xp= np.zeros_like(X)
        for t in range(I):
            dlt = model.predict(Xp) - Y
            dir = np.sign(np.matmul(dlt, model.weight.T))
            Xp = Xp + epsilon * dir
            Xp = np.where(Xp > X+epsilon, X+epsilon, Xp)
            Xp = np.where(Xp < X-epsilon, X-epsilon, Xp)
        return Xp, Y


