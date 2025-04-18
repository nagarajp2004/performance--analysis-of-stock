import yfinance as yf
import pandas as pd
from prophet import Prophet

# Fetch stock data
def fetch_stock_data(stock_code):
    stock_data = yf.download(stock_code, start='2015-01-01')
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    return stock_data

# Long-term forecast using Prophet
def long_term_forecast(stock_data):
    prophet_df = stock_data[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df.loc[:, 'ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

if __name__ == "__main__":
    stock_code = 'AAPL'
    data = fetch_stock_data(stock_code)
    long_term = long_term_forecast(data)
    print(f"Long-term forecast: \n{long_term.tail()}")
