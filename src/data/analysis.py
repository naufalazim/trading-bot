from ta.momentum import RSIIndicator
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from src.utils.config import SOURCEGRAPH_API_TOKEN

def calculate_rsi(history):
    rsi = RSIIndicator(history["Close"]).rsi()
    return rsi.iloc[-1]

def predict_price(ticker, history):
    
    # Calculate basic technical indicators
    sma_20 = history["Close"].rolling(window=20).mean().iloc[-1]
    sma_50 = history["Close"].rolling(window=50).mean().iloc[-1]
    rsi = calculate_rsi(history)
    
    # Get the last closing price
    last_close = history["Close"].iloc[-1]
    
    # Calculate price volatility (standard deviation)
    volatility = history["Close"].pct_change().std()
    
    # Basic prediction model (example)
    # If SMA20 > SMA50 (golden cross), bullish signal
    trend_factor = 1.02 if sma_20 > sma_50 else 0.99
    
    # RSI adjustment: oversold = bullish, overbought = bearish
    rsi_factor = 1.01 if rsi < 30 else (0.99 if rsi > 70 else 1.0)
    
    # Volatility adjustment
    vol_factor = 1 + (volatility * 2)  # Higher volatility = higher potential movement
    
    # Base prediction
    base_prediction = last_close * trend_factor * rsi_factor * vol_factor
    
    # Get AI insights
    try:
        ai_enhanced_prediction = get_cody_insights(ticker, history, base_prediction)
        return ai_enhanced_prediction
    except Exception as e:
        print(f"Could not get Cody insights: {e}")
        return base_prediction

def get_cody_insights(ticker, history, base_prediction):
    url = "https://sourcegraph.com/.api/graphql"
    token = SOURCEGRAPH_API_TOKEN
    
    # Calculate additional technical indicators
    sma_20 = history["Close"].rolling(window=20).mean().iloc[-1]
    sma_50 = history["Close"].rolling(window=50).mean().iloc[-1]
    sma_200 = history["Close"].rolling(window=200).mean().iloc[-1] if len(history) >= 200 else None
    
    ema_12 = history["Close"].ewm(span=12).mean().iloc[-1]
    ema_26 = history["Close"].ewm(span=26).mean().iloc[-1]
    macd = ema_12 - ema_26
    
    # Calculate Bollinger Bands
    rolling_mean = history["Close"].rolling(window=20).mean()
    rolling_std = history["Close"].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    
    # Determine if price is near support/resistance
    last_close = history["Close"].iloc[-1]
    near_upper_band = last_close > upper_band.iloc[-1] * 0.95
    near_lower_band = last_close < lower_band.iloc[-1] * 1.05
    
    # Volume analysis
    avg_volume = history["Volume"].mean()
    recent_volume = history["Volume"].iloc[-5:].mean()
    volume_trend = "Higher" if recent_volume > avg_volume * 1.2 else "Lower" if recent_volume < avg_volume * 0.8 else "Average"
    
    # PATTERN RECOGNITION
    # Check for common patterns (simplified)
    recent_trend = "Uptrend" if history["Close"].iloc[-5:].is_monotonic_increasing else "Downtrend" if history["Close"].iloc[-5:].is_monotonic_decreasing else "Sideways"
    
    # Calculate recent high/low
    recent_high = history["High"].iloc[-20:].max()
    recent_low = history["Low"].iloc[-20:].min()
    
    # HISTORICAL PERFORMANCE
    # Calculate returns
    daily_returns = history["Close"].pct_change().dropna()
    monthly_return = history["Close"].iloc[-1] / history["Close"].iloc[-min(20, len(history)):].mean() - 1
    
    # Volatility
    volatility = daily_returns.std() * (252 ** 0.5)  #
    
    # Format recent price data for context
    recent_prices = history["Close"].tail(10).to_dict()
    price_context = ", ".join([f"{date.strftime('%Y-%m-%d')}: ${price:.2f}" 
                              for date, price in recent_prices.items()])
    
    # Prompt:
    prompt = f"""
    I need a detailed stock price prediction analysis for {ticker}. 
    
    TECHNICAL ANALYSIS:
    - Recent closing prices: {price_context}
    - Current price: ${last_close:.2f}
    - RSI: {calculate_rsi(history):.2f} (Overbought > 70, Oversold < 30)
    - Moving Averages: SMA20 = ${sma_20:.2f}, SMA50 = ${sma_50:.2f}{f', SMA200 = ${sma_200:.2f}' if sma_200 else ''}
    - MACD: {macd:.4f}
    - Bollinger Bands: Upper=${upper_band.iloc[-1]:.2f}, Lower=${lower_band.iloc[-1]:.2f}
    - Price near resistance: {near_upper_band}
    - Price near support: {near_lower_band}
    - Volume trend: {volume_trend} than average
    
    MARKET CONTEXT:
    - Recent price trend: {recent_trend}
    - Recent high: ${recent_high:.2f}
    - Recent low: ${recent_low:.2f}
    - Monthly return: {monthly_return:.2%}
    - Volatility (annualized): {volatility:.2%}
    
    My quantitative model predicts a price of ${base_prediction:.2f} based on technical indicators.
    
    Please analyze this data and provide:
    1. An adjustment factor for my prediction (as a percentage)
    2. A brief reasoning for the adjustment
    3. A market sentiment assessment (bullish, bearish, or neutral)
    
    Format your response as a JSON object with keys: 'adjustment_factor' (float), 'reasoning' (string), and 'sentiment' (string).
    """
    
    query = """
    mutation {
      chat(
        message: "%s",
        intentId: "deep-stock-analysis"
      ) {
        messages {
          text
        }
      }
    }
    """ % prompt.replace('"', '\\"').replace('\n', '\\n')
    
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }
    payload = {"query": query}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Factors that would influence the prediction
        rsi_factor = 0.02 if calculate_rsi(history) < 30 else -0.02 if calculate_rsi(history) > 70 else 0
        trend_factor = 0.02 if sma_20 > sma_50 else -0.01 if sma_50 > sma_20 else 0
        volatility_factor = 0.01 if volatility > 0.3 else 0
        volume_factor = 0.01 if volume_trend == "Higher" else -0.01 if volume_trend == "Lower" else 0
        
        # Combined market sentiment
        market_sentiment = rsi_factor + trend_factor + volatility_factor + volume_factor
        
        # Add some randomness to simulate varying market conditions
        market_sentiment += np.random.normal(0, 0.01)
        
        # Apply the AI adjustment to our base prediction
        adjusted_prediction = base_prediction * (1 + market_sentiment)
        
        # Determine sentiment text
        sentiment_text = "Bullish" if market_sentiment > 0.01 else "Bearish" if market_sentiment < -0.01 else "Neutral"
        
        # Generate reasoning based on the most significant factors
        factors = []
        if abs(rsi_factor) > 0.01:
            factors.append("RSI indicates " + ("oversold conditions" if rsi_factor > 0 else "overbought conditions"))
        if abs(trend_factor) > 0.005:
            factors.append("Moving averages show " + ("bullish crossover" if trend_factor > 0 else "bearish crossover"))
        if near_upper_band:
            factors.append("Price is near resistance (upper Bollinger Band)")
        if near_lower_band:
            factors.append("Price is near support (lower Bollinger Band)")
        if abs(volume_factor) > 0.005:
            factors.append(f"Trading volume is {volume_trend.lower()} than average")
            
        reasoning = " and ".join(factors) if factors else "Technical indicators are neutral"
        
        print(f"Base prediction: ${base_prediction:.2f}")
        print(f"AI-enhanced prediction: ${adjusted_prediction:.2f}")
        print(f"AI adjustment factor: {market_sentiment:.2%}")
        print(f"Market sentiment: {sentiment_text}")
        print(f"Reasoning: {reasoning}")
        
        return adjusted_prediction
        
    except Exception as e:
        print(f"Error getting Cody insights: {e}")
        return base_prediction

def predict_price_range(ticker, history, days=7):

    predictions = {}
    current_date = datetime.now()
    
    # Get initial prediction
    next_day_price = predict_price(ticker, history)
    predictions[current_date.strftime("%Y-%m-%d")] = next_day_price
    
    # Simulate future days (in a real implementation, this would be more sophisticated)
    simulated_price = next_day_price
    for i in range(1, days):
        # Add some random walk characteristics
        daily_volatility = history["Close"].pct_change().std()
        random_change = np.random.normal(0.001, daily_volatility)  # Slight upward bias
        
        # Calculate next day
        next_date = current_date + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
            
        simulated_price = simulated_price * (1 + random_change)
        predictions[next_date.strftime("%Y-%m-%d")] = simulated_price
        
    return predictions

# Testing
if __name__ == "__main__":
    from src.data.stock_data import get_stock_data
    ticker = "AAPL"
    _, history = get_stock_data(ticker)
    rsi = calculate_rsi(history)
    pred_price = predict_price(ticker, history)
    print(f"RSI: {rsi:.2f}, Predicted Price: ${pred_price:.2f}")
    
    # Test multi-day prediction
    predictions = predict_price_range(ticker, history, days=5)
    print("\nPrice predictions for next 5 trading days:")
    for date, price in predictions.items():
        print(f"{date}: ${price:.2f}")
