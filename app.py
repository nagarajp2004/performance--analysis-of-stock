import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go

# App Title
st.title('Stock Market Prediction App')
st.sidebar.header('User Input Parameters')

# Function to fetch stock data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# User input for stock ticker and date range
ticker = st.sidebar.text_input('Enter Stock Code', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

# Load and display the data
data = load_data(ticker, start_date, end_date)

if data.empty:
    st.warning("No data available. Please try a different stock code or date range.")
else:
    st.write(f'Showing data for {ticker}')
    st.dataframe(data.tail())

    # CSV Download for Raw Data
    csv_data = data.to_csv(index=False)
    st.download_button(
        label="Download Raw Data as CSV",
        data=csv_data,
        file_name=f'{ticker}_historical_data.csv',
        mime='text/csv'
    )

    # Preparing data for Prophet
    prophet_df = data[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

    # Prophet Model
    model = Prophet()
    model.fit(prophet_df)

    # Forecast for future days (user-defined)
    forecast_days = st.sidebar.slider('Select Forecast Days', min_value=30, max_value=365, step=30, value=365)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Forecast Plot
    st.subheader('Long-term Forecast')
    fig1 = go.Figure([
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dot')),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dot')),
        go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='black'))
    ])
    fig1.update_layout(title="Stock Price Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig1)

    # CSV Download for Forecast Data
    forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
    st.download_button(
        label="Download Forecast Data as CSV",
        data=forecast_csv,
        file_name=f'{ticker}_forecast_data.csv',
        mime='text/csv'
    )
