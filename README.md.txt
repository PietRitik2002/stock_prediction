Stock Price Prediction with Trigger

A Streamlit web application for predicting stock prices using LSTM models in PyTorch and TensorFlow, with a price trigger feature to alert users when predictions cross a threshold.

Features





Fetches historical stock data using yfinance.



Trains LSTM models with PyTorch or TensorFlow.



Visualizes actual vs. predicted prices using Plotly.



Alerts users if the latest predicted price crosses a user-defined threshold.

Installation





Clone the repository:

git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction



Install dependencies:

pip install -r requirements.txt



Run the Streamlit app:

streamlit run stock_prediction_app_with_trigger.py

Usage





Open the app at http://localhost:8501.



Enter a stock ticker (e.g., AAPL).



Select a model (TensorFlow or PyTorch).



Set a price trigger threshold (e.g., $150).



Click "Predict" to view results and alerts.

Requirements





Python 3.8+



Libraries: yfinance, pandas, numpy, scikit-learn, tensorflow, torch, streamlit, plotly