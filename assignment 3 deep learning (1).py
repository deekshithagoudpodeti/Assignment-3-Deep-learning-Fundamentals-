#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import seaborn as sns
import tensorflow as tf


# ## Step-1 Reading the DataSet and Visualizing 

# In[100]:


import pandas as pd

# Loading the datasets
train_data = pd.read_csv(r"C:\Users\podeti venu goud\Downloads\Google_Stock_Price_Train.csv", thousands=',')
test_data = pd.read_csv(r"C:\Users\podeti venu goud\Downloads\Google_Stock_Price_Test.csv", thousands=',')

# Displaying basic information about the datasets
print("Train Data Summary:")
print(train_data.describe())

print("\nTest Data Summary:")
print(test_data.describe())


# In[101]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Loading the Data
train_data = pd.read_csv(r"C:\Users\podeti venu goud\Downloads\Google_Stock_Price_Train.csv", thousands=',')
test_data = pd.read_csv(r"C:\Users\podeti venu goud\Downloads\Google_Stock_Price_Test.csv", thousands=',')

# Combine both datasets into a single DataFrame
true_df = pd.concat([train_data, test_data], axis=0)

# Step 2: Preprocessing the Data
# Dropping rows with missing values
true_df = true_df.dropna()

# Dropping duplicates
true_df = true_df.drop_duplicates()

# Resetting the index
true_df.reset_index(drop=True, inplace=True)

# Checking the column names
print("Columns in the dataset:")
print(true_df.columns.values)

# Step 3: Train & Test Definition
# Splitting the data into training and testing sets (90% train, 10% test)
train = true_df[:int(0.8 * len(true_df))] 
test = true_df.iloc[int(0.8 * len(true_df)):, :]  

# Displaying a preview of the training set
print("\nTraining Set:")
print(train.head())

# Displaying a preview of the test set
print("\nTesting Set:")
print(test.head())

# Step 4: Visualization
# Plotting histograms of the dataset
true_df.hist(figsize=(15, 12), color='blue')
plt.suptitle('Distribution of Google Stock Price Data', fontsize=16)
plt.show()


# ## Step-2 Bulding the RNN Model

# In[87]:


# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense

# Step 2: Reading and Preprocessing the Dataset
train_data = pd.read_csv(r"C:\Users\podeti venu goud\Downloads\Google_Stock_Price_Train.csv", thousands=',')
test_data = pd.read_csv(r"C:\Users\podeti venu goud\Downloads\Google_Stock_Price_Test.csv", thousands=',')

# Dropping rows with missing values and duplicates
train_data = train_data.dropna()
train_data = train_data.drop_duplicates()
test_data = test_data.dropna()
test_data = test_data.drop_duplicates()

# Resetting index
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Selecting 'Close' column for training and test
train_close = train_data.iloc[:, 4:5].values  # 'Close' is at index 4
test_close = test_data.iloc[:, 4:5].values

# Step 3: Normalizing the Data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = scaler.fit_transform(train_close)
test_scaled = scaler.transform(test_close)

# Step 4: Creating Sequences for Training
# Using past 60 days data to predict the next day's price
X_train = []
y_train = []

for i in range(60, len(train_scaled)):
    X_train.append(train_scaled[i-60:i, 0])  # 60 previous days' stock prices
    y_train.append(train_scaled[i, 0])  # next day's stock price

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data to be compatible with the RNN input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Step 5: Building the RNN Model
rnn_model = Sequential()

# Adding RNN layers with dropout regularization
rnn_model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(units=50))
rnn_model.add(Dropout(0.2))

# Output layer
rnn_model.add(Dense(units=1))

# Compiling the RNN
rnn_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])

# Step 6: Training the Model
rnn_history = rnn_model.fit(X_train, y_train, epochs=100, batch_size=32)

# Preparing Test Data
total_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
total_data.reset_index(drop=True, inplace=True)

# Preparing inputs for prediction from the last 60 days of data
inputs = total_data[len(total_data) - len(test_data) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

# Creating test sequences
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 7: Making Predictions
predicted_stock_price = rnn_model.predict(X_test)

# Inverse transforming predictions
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Results
print("Predicted Stock Prices:", predicted_stock_price[:5])  # Display first 5 predictions


# In[88]:


# Step 8: Evaluating the Model

# Real stock prices for test data
real_stock_price = test_close

# R2 Score
from sklearn.metrics import r2_score
r2 = r2_score(real_stock_price, predicted_stock_price)
print(f"R2 Score: {r2}")

# RMSE
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(f"RMSE: {rmse}")


# ## Visualizing the RNN Model

# In[89]:


# Step 9: Plotting the Real vs Predicted Stock Prices

import matplotlib.pyplot as plt

# Plotting real stock prices vs predicted stock prices
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction with RNN')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# ## Step-3 Building LSTM Model

# In[106]:


# LSTM Model
from tensorflow.keras.layers import LSTM

# Step 5: Building the LSTM Model
lstm_model = Sequential()

# Adding LSTM layers with dropout regularization
lstm_model.add(LSTM(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(units=50, activation="tanh", return_sequences=True))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(units=50, activation="tanh"))
lstm_model.add(Dropout(0.2))

# Output layer
lstm_model.add(Dense(units=1))

# Compiling the LSTM
lstm_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])

# Step 6: Training the LSTM Model
lstm_history = lstm_model.fit(X_train, y_train, epochs=100, batch_size=32)

# Making Predictions
lstm_predicted_price = lstm_model.predict(X_test)

# Inverse transforming predictions
lstm_predicted_price = scaler.inverse_transform(lstm_predicted_price)

# Results
print("Predicted Stock Prices (LSTM):", lstm_predicted_price[:5])  # Display first 5 predictions


# ## Step -4 Building GRU Model

# In[90]:


# GRU Model
from tensorflow.keras.layers import GRU

# Step 5: Building the GRU Model
gru_model = Sequential()

# Adding GRU layers with dropout regularization
gru_model.add(GRU(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
gru_model.add(Dropout(0.2))

gru_model.add(GRU(units=50, activation="tanh", return_sequences=True))
gru_model.add(Dropout(0.2))

gru_model.add(GRU(units=50, activation="tanh"))
gru_model.add(Dropout(0.2))

# Output layer
gru_model.add(Dense(units=1))

# Compiling the GRU model
gru_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])

# Step 6: Training the GRU Model
gru_history = gru_model.fit(X_train, y_train, epochs=100, batch_size=32)

# Making Predictions
gru_predicted_price = gru_model.predict(X_test)

# Inverse transforming predictions
gru_predicted_price = scaler.inverse_transform(gru_predicted_price)

# Results
print("Predicted Stock Prices (GRU):", gru_predicted_price[:5])  # Display first 5 predictions


# ## Step-5 Evaluating All Models

# In[107]:


# Step 9: Evaluating All Models

# Real stock prices for test data
real_stock_price = test_close

# R2 Score for RNN
from sklearn.metrics import r2_score
rnn_r2 = r2_score(real_stock_price, predicted_stock_price)
print(f"RNN R2 Score: {rnn_r2}")

# RMSE for RNN
rnn_rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(f"RNN RMSE: {rnn_rmse}")

# R2 Score for LSTM
lstm_r2 = r2_score(real_stock_price, lstm_predicted_price)
print(f"LSTM R2 Score: {lstm_r2}")

# RMSE for LSTM
lstm_rmse = sqrt(mean_squared_error(real_stock_price, lstm_predicted_price))
print(f"LSTM RMSE: {lstm_rmse}")

# R2 Score for GRU
gru_r2 = r2_score(real_stock_price, gru_predicted_price)
print(f"GRU R2 Score: {gru_r2}")

# RMSE for GRU
gru_rmse = sqrt(mean_squared_error(real_stock_price, gru_predicted_price))
print(f"GRU RMSE: {gru_rmse}")


# ## Step- 6 Plotting the Models

# In[108]:


# Step 10: Plotting the Real vs Predicted Stock Prices for Each Model

# Plotting Real vs Predicted for RNN
plt.figure(figsize=(15, 5))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price (RNN)')
plt.title('Google Stock Price Prediction with RNN')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Plotting Real vs Predicted for LSTM
plt.figure(figsize=(15, 5))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(lstm_predicted_price, color='green', label='Predicted Google Stock Price (LSTM)')
plt.title('Google Stock Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Plotting Real vs Predicted for GRU
plt.figure(figsize=(15, 5))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(gru_predicted_price, color='purple', label='Predicted Google Stock Price (GRU)')
plt.title('Google Stock Price Prediction with GRU')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# ## Step-7 Checking RMSE and R2 Score's 

# In[109]:


# Step 9: Evaluating All Models

# Real stock prices for test data
real_stock_price = test_close

# R2 Score for RNN
from sklearn.metrics import r2_score
rnn_r2 = r2_score(real_stock_price, predicted_stock_price)
print(f"RNN R2 Score: {rnn_r2}")

# RMSE for RNN
rnn_rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(f"RNN RMSE: {rnn_rmse}")

# R2 Score for LSTM
lstm_r2 = r2_score(real_stock_price, lstm_predicted_price)
print(f"LSTM R2 Score: {lstm_r2}")

# RMSE for LSTM
lstm_rmse = sqrt(mean_squared_error(real_stock_price, lstm_predicted_price))
print(f"LSTM RMSE: {lstm_rmse}")

# R2 Score for GRU
gru_r2 = r2_score(real_stock_price, gru_predicted_price)
print(f"GRU R2 Score: {gru_r2}")

# RMSE for GRU
gru_rmse = sqrt(mean_squared_error(real_stock_price, gru_predicted_price))
print(f"GRU RMSE: {gru_rmse}")


# ## Step-8 Checking for MSE and MAPE

# In[110]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# MAPE calculation function
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# In[111]:


# Assuming the following variables are already defined:
# predicted_stock_price: RNN predictions
# lstm_predicted_price: LSTM predictions
# gru_predicted_price: GRU predictions
# real_stock_price: The actual stock prices (test data)

# Step 3: Calculate MSE, MAPE, and R2 for each model

# For RNN Model
rnn_mse = mean_squared_error(real_stock_price, predicted_stock_price)
rnn_mape = mean_absolute_percentage_error(real_stock_price, predicted_stock_price)
rnn_r2 = r2_score(real_stock_price, predicted_stock_price)

print(f"RNN MSE: {rnn_mse}")
print(f"RNN MAPE: {rnn_mape}%")
print(f"RNN R2 Score: {rnn_r2}")

# For LSTM Model
lstm_mse = mean_squared_error(real_stock_price, lstm_predicted_price)
lstm_mape = mean_absolute_percentage_error(real_stock_price, lstm_predicted_price)
lstm_r2 = r2_score(real_stock_price, lstm_predicted_price)

print(f"LSTM MSE: {lstm_mse}")
print(f"LSTM MAPE: {lstm_mape}%")
print(f"LSTM R2 Score: {lstm_r2}")

# For GRU Model
gru_mse = mean_squared_error(real_stock_price, gru_predicted_price)
gru_mape = mean_absolute_percentage_error(real_stock_price, gru_predicted_price)
gru_r2 = r2_score(real_stock_price, gru_predicted_price)

print(f"GRU MSE: {gru_mse}")
print(f"GRU MAPE: {gru_mape}%")
print(f"GRU R2 Score: {gru_r2}")

