import requests
import pandas as pd
import time

def fetch_single_stock(symbol, start, end, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'results' in data:
        df = pd.DataFrame(data['results'])
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        df['Adj Close'] = df['Close']  # Assume Close == Adj Close for simplicity
        return df
    else:
        print(f"No data available for {symbol}")
        return pd.DataFrame()  # Return an empty DataFrame if no data is available

def fetch_data_for_multiple_stocks(tickers, start, end, api_key):
    all_data = {}
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = fetch_single_stock(ticker, start, end, api_key)
        if not df.empty:
            all_data[ticker] = df
        time.sleep(0.5)  # Delay to avoid hitting rate limits
    
    # Combine all dataframes
    if all_data:
        combined_df = pd.concat(all_data, axis=1)
        combined_df.columns.names = ['Symbol', 'Feature']
        return combined_df
    else:
        print("No data available for any of the provided symbols.")
        return pd.DataFrame()
