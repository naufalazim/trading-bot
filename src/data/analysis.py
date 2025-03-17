from ta.momentum import RSIIndicator
import pandas as pd
import requests
from src.utils import config  # Assuming config.py has API keys

def calculate_rsi(history):
    rsi = RSIIndicator(history["Close"]).rsi()
    return rsi.iloc[-1]

def predict_price(ticker, history):
    url = "https://sourcegraph.com/.api/graphql"
    token = "SOURCEGRAPH_API_TOKEN"  # Token
    
    # Hypothetical GraphQL query to ask Cody for prediction logic
    # Note: Cody doesn't have a public API for this yet, so this is a placeholder
    query = """
    query {
      currentUser {
        username
      }
      # Simulated Cody response (not real API functionality)
      codyPrediction(ticker: "%s") {
        predictedPrice
      }
    }
    """ % ticker
    
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }
    payload = {"query": query}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Mock response since Cody doesn’t support this directly
        # In reality, you'd need Cody to return prediction logic or fetch it from a repo
        if "data" in data and "codyPrediction" in data["data"]:
            return data["data"]["codyPrediction"]["predictedPrice"]
        else:
            # Fallback: Simple average-based prediction (simulating Cody’s "suggestion")
            return history["Close"].mean() * 1.05  # 5% increase as a placeholder
    except requests.RequestException as e:
        print(f"Error with Sourcegraph API: {e}")
        # Fallback to basic prediction
        return history["Close"].mean() * 1.05

# Example usage (for testing)
if __name__ == "__main__":
    from src.data.stock_data import get_stock_data
    ticker = "AAPL"
    _, history = get_stock_data(ticker)
    rsi = calculate_rsi(history)
    pred_price = predict_price(ticker, history)
    print(f"RSI: {rsi:.2f}, Predicted Price: ${pred_price:.2f}")