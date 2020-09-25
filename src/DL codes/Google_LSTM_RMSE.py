import numpy as np
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import math
#matplotlib inline

from keras import Sequential
from keras.layers import Dense, LSTM


from sklearn.preprocessing import MinMaxScaler



# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

# define path to save model
model_path = '../../Output/Google_regression_CNN.h5'

# import data
stock_data = pd.read_csv("../../Dataset/Google_train.csv",dtype={'Close': 'float64', 'Volume': 'int64','Open': 'float64','High': 'float64', 'Low': 'float64'})
print(stock_data.head(2))
stock_data.info()

#name the columns
stock_data.columns = ['date', 'close/last', 'volume', 'open', 'high', 'low']

stock_data.info()

#create a new column "average" 
stock_data['average'] = (stock_data['high'] + stock_data['low'])/2

stock_data.info()
print(stock_data.head(2))

#pick the input features (average and volume)
input_feature= stock_data.iloc[:,[2,6]].values
input_data = input_feature

#data normalization
sc= MinMaxScaler(feature_range=(0,1))
input_data[:,0:2] = sc.fit_transform(input_feature[:,:])


# data preparation
lookback = 60

test_size = int(.3 * len(stock_data))
X = []
y = []
for i in range(len(stock_data) - lookback - 1):
    t = []
    for j in range(0, lookback):
        t.append(input_data[[(i + j)], :])
    X.append(t)
    y.append(input_data[i + lookback, 1])


X, y= np.array(X), np.array(y)
X_test = X[:test_size+lookback]

print(X_test.shape)
print(X.shape)

X = X.reshape(X.shape[0],lookback, 2)
X_test = X_test.reshape(X_test.shape[0],lookback, 2)
print(X.shape)
print(X_test.shape)
print(y.shape)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model = Sequential()
model.add(LSTM(
         input_shape=(X.shape[1], 2),
         units=30,
         return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
          units=30,
          return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
          units=30,
          return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse])
print(model.summary())


# fit the network
history = model.fit(X, y, epochs=200, batch_size=32, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())


scores= model.predict(X_test)
print('\nRMSE: {}'.format(scores[1]))

plt.plot(scores, color= 'red')
plt.plot(input_data[lookback:test_size+(2*lookback),1], color='green')
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()