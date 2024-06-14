import os
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Set the time zone environment variable
os.environ['TZ'] = 'UTC'

# Define constants
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title('Stock Trend Forecasting')

# Dropdown for selecting stock ticker
stocks = ('GOOG', 'AAPL', 'MSFT', 'META', 'AMZN')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider for selecting number of years to predict
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    # Ensure the index is a DatetimeIndex
    if not isinstance(data['Date'], pd.DatetimeIndex):
        data['Date'] = pd.to_datetime(data['Date'])
    return data

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data function
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Plot raw data
plot_raw_data()

# Prepare data for forecasting
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = df_train['ds'].dt.date

# Train the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe and make predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
