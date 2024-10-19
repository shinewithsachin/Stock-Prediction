# Importing necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM # type: ignore
from keras.models import load_model #type: ignore

# 1. Generate Example Stock Data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100)
stock_prices = np.random.rand(100) * 100  # Random stock prices between 0 and 100

# Create a DataFrame to visualize the data
stock_data = pd.DataFrame({'Date': dates, 'Close': stock_prices})
print("Sample Stock Data:")
print(stock_data.head())

# 2. Prepare Data for Model (Use only the 'Close' prices)
X = stock_prices[:-1]  # Previous day's stock price
y = stock_prices[1:]   # Next day's stock price

# Reshape the data to fit the neural network
X = X.reshape(-1, 1)  # Shape (n_samples, 1)
y = y.reshape(-1, 1)  # Shape (n_samples, 1)

X = X.reshape(X.shape[0], 1, 1)

# 3. Build a Simple Neural Network Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, 1)),  # Input layer
    Dense(64, activation='relu'),                      # Hidden layer
    Dense(1)                                           # Output layer for predicting stock price
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (50 epochs)
print("Training the model...")
model.fit(X, y, epochs=50)

# 4. Save the Model in .keras Format
model.save('Stock_Predictions_Model.keras')
print("Model saved successfully as 'Stock_Predictions_Model.keras'.")

# 5. Load the Model
print("Loading the model...")
loaded_model = load_model('Stock_Predictions_Model.keras')
print("Model loaded successfully.")

# 6. Make a Prediction
# Predicting the stock price for the next day based on the last known day's price
last_day_price = np.array([[stock_prices[-1]]])  # Last known stock price
last_day_price = last_day_price.reshape(1, 1, 1)
predicted_price = loaded_model.predict(last_day_price)
print(f"Predicted next day price: {predicted_price[0][0]:.2f}")
