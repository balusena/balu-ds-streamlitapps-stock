import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import logging

# Set logging configuration
logging.basicConfig(level=logging.DEBUG)

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Forecasting')

stocks = ('GOOG', 'AAPL', 'MSFT', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logging.error(f"Error loading data for {ticker}: {e}")
        return None


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
if data is not None:
    data_load_state.text('Loading data... done!')
else:
    st.error("Failed to load data. Please check the selected stock symbol.")

st.subheader('Raw data')
st.write(data.tail() if data is not None else "No data to display.")


# Plot raw data
def plot_raw_data():
    if data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.warning("No data available to plot.")


plot_raw_data()

# Predict forecast with Prophet.
if data is not None:
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
else:
    st.error("Forecasting cannot be performed due to missing data.")
