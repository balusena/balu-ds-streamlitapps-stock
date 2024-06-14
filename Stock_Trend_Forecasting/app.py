import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")

stocks = ("ZOMATO.NS","VEDL.NS","IRFC.NS","RVNL.NS", "HINDALCO.NS","JSWSTEEL.NS","GRASIM.NS","ULTRACEMCO.NS","BAJAJ-AUTO.NS","IRCTC.NS","GC=F", "AAPL", "TSLA","ADANIENT.NS","ADANIPORTS.NS","ADANIPOWER.NS","ADANIGREEN.NS","ATGL.NS")
selected_stocks = st.selectbox("Select Stock", stocks)

n_years = st.slider("Years of Prediction", 0.1, 15.0,step=0.1)
period = int(n_years * 365.0)


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data....")
data = load_data(selected_stocks)
data_load_state.text("Loding data...done!")

st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write(selected_stocks, 'Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
