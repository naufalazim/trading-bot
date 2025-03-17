import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def get_stock_data(ticker, period="1mo", interval="1d", retries=3):
    """
    Get stock data:
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        retries (int): Number of retry attempts if API call fails
    
    Returns:
        tuple: (current_price, history_dataframe)
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, interval=interval)
            
            # Handle empty data
            if history.empty:
                print(f"Warning: No data returned for {ticker}")
                return 0.0, pd.DataFrame()
            
            # Get current price (last closing price)
            current_price = history["Close"].iloc[-1]
            
            # Add technical indicators
            history = add_technical_indicators(history)
            
            return current_price, history
            
        except Exception as e:
            if attempt < retries - 1:
                print(f"Error fetching data for {ticker}, retrying... ({e})")
                time.sleep(2)  # Wait before retrying
            else:
                print(f"Failed to fetch data for {ticker} after {retries} attempts: {e}")
                return 0.0, pd.DataFrame()

def add_technical_indicators(df):
    """Add common technical indicators to the dataframe"""
    # Ensure we have enough data
    if len(df) < 20:
        return df
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    stddev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (stddev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (stddev * 2)
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    return df

def get_historical_data(ticker, start_date, end_date=None):
    """
    Get historical stock data for a specific date range
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
    
    Returns:
        DataFrame: Historical stock data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        return add_technical_indicators(history)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

def get_multiple_stocks_data(tickers, period="1mo"):
    """
    Get data for multiple stocks at once
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Data period
    
    Returns:
        dict: Dictionary of {ticker: (current_price, history)}
    """
    results = {}
    for ticker in tickers:
        current_price, history = get_stock_data(ticker, period)
        results[ticker] = (current_price, history)
    return results

def get_market_index_data(index_symbol="^GSPC", period="1mo"):
    """
    Get market index data (default: S&P 500)
    
    Args:
        index_symbol (str): Index symbol (^GSPC for S&P 500, ^DJI for Dow Jones, ^IXIC for NASDAQ)
        period (str): Data period
    
    Returns:
        tuple: (current_value, history_dataframe)
    """
    return get_stock_data(index_symbol, period)

# TESTING
if __name__ == "__main__":
    # Test single stock
    ticker = "AAPL"
    current_price, history = get_stock_data(ticker)
    print(f"{ticker} current price: ${current_price:.2f}")
    print(f"Available data points: {len(history)}")
    print(f"Available columns: {history.columns.tolist()}")
    
    # Test multiple stocks
    tickers = ["AAPL", "MSFT", "GOOGL"]
    multi_data = get_multiple_stocks_data(tickers)
    for tick, (price, _) in multi_data.items():
        print(f"{tick}: ${price:.2f}")
    
    # Test market index
    index_price, index_history = get_market_index_data()
    print(f"S&P 500 current value: {index_price:.2f}")
