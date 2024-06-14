# Install necessary packages in your environment
# pip install streamlit prophet yfinance plotly

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set up Streamlit title and date variables
st.title('Stock Trend Forecasting')
TODAY = date.today().strftime("%Y-%m-%d")

# Stock symbols for selection
stocks = ('GOOG', 'AAPL', 'MSFT', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider for selecting number of years to forecast
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365  # Convert years into days

# Function to load data from Yahoo Finance using yfinance
@st.cache_data  # Cache data loading process to optimize performance
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end=TODAY, progress=False)
    data.reset_index(inplace=True)
    return data

# Load data for the selected stock symbol
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data using Plotly
st.subheader('Raw data plot')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Forecasting with Prophet
st.subheader('Forecasting')

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Create Prophet model and fit data
m = Prophet()
m.fit(df_train)

# Make future dataframe for forecasting
future = m.make_future_dataframe(periods=period)

# Forecasting
forecast = m.predict(future)

# Display forecast data
st.write('Forecast data')
st.write(forecast.tail())

# Plot forecast
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
