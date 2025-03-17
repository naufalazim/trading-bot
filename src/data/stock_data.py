import yfinance as yf

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="1d")["Close"].iloc[-1]
    history = stock.history(period="1mo")  
    return current_price, history