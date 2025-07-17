import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Streamlit page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Data fetching and preprocessing
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].values

def preprocess_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# TensorFlow LSTM Model
def build_tf_model(lookback):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# PyTorch LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.fc2(out)
        return out

def train_pytorch_model(model, X_train, y_train, epochs=50, batch_size=32):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Prediction function
def predict_stock_price(ticker, model_type='tensorflow'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = fetch_stock_data(ticker, start_date, end_date)
    
    lookback = 60
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, lookback)
    
    if model_type == 'tensorflow':
        model = build_tf_model(lookback)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        predictions = model.predict(X_test, verbose=0)
    else:
        model = StockLSTM()
        train_pytorch_model(model, X_train, y_train)
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            predictions = model(X_test_tensor).numpy()
    
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return actual, predictions

# Streamlit UI
def main():
    st.title("Stock Price Prediction")
    st.write("Enter a stock ticker and select a model to predict future stock prices.")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
    with col2:
        model_type = st.selectbox("Model Type", ["TensorFlow", "PyTorch"])

    if st.button("Predict"):
        with st.spinner("Fetching data and training model..."):
            try:
                model_type_lower = 'tensorflow' if model_type == 'TensorFlow' else 'pytorch'
                actual, predictions = predict_stock_price(ticker, model_type_lower)
                
                # Create Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=actual.flatten(), name='Actual', line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(y=predictions.flatten(), name='Predicted', line=dict(color='#ff7f0e')))
                fig.update_layout(
                    title=f"{ticker} Stock Price Prediction",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()