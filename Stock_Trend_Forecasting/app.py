import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import logging

# Set up Streamlit title and date variables
st.title('Stock Trend Forecasting')
TODAY = date.today().strftime("%Y-%m-%d")

# Stock symbols for selection
stocks = ('GOOG', 'AAPL', 'MSFT', 'META', 'AMZN')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider for selecting number of years to forecast
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365  # Convert years into days

# Function to load data from Yahoo Finance using yfinance
@st.cache
def load_data(ticker):
    try:
        # Dynamically adjust START date based on the earliest available date for the stock symbol
        start_date = max(pd.Timestamp('2010-01-01'), yf.Ticker(ticker).history(period="max").index.min())
        data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=TODAY, progress=False)
        data.reset_index(inplace=True)
        
        # Rename columns to avoid duplicates and ensure consistency
        data.rename(columns={"Date": "ds", "Open": "open", "High": "high", "Low": "low", 
                             "Close": "y", "Adj Close": "adj_close", "Volume": "volume"}, inplace=True)
        
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        logging.error(f"Error loading data for {ticker}: {e}")
        return None

# Handle exceptions during data loading and processing
try:
    # Load data for the selected stock symbol
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    if data is not None and not data.empty:
        data_load_state.text('Loading data... done!')
        st.write(data.head())  # Inspect the first few rows for debugging
    else:
        data_load_state.text('Loading data... failed!')

    # Display raw data
    st.subheader('Raw data')
    if data is not None and not data.empty:
        st.write(data.tail())
    else:
        st.error("No data available to display.")

    # Plot raw data using Plotly
    st.subheader('Raw data plot')
    if data is not None and not data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['open'], mode='lines', name='Open'))
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Close'))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.error("No data available to plot.")

    # Forecasting with Prophet
    st.subheader('Forecasting')

    if data is not None and not data.empty:
        # Prepare data for Prophet
        df_train = data[['ds', 'y']].copy()

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
    else:
        st.error("Forecasting cannot be performed due to missing data.")

except Exception as e:
    st.error(f"Error occurred: {str(e)}")
    logging.error(f"Error occurred: {e}")
