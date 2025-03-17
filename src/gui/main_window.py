from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QLabel, 
                              QComboBox, QPushButton, QHBoxLayout, QGridLayout, QFrame)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta
from src.data.stock_data import get_stock_data
from src.data.analysis import calculate_rsi, predict_price, predict_price_range

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NopalGPT Stock Analysis")
        self.setGeometry(100, 100, 1000, 800)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create header with stock selection
        header_layout = QHBoxLayout()
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "MAYBANK"])
        self.ticker_combo.setCurrentText("AAPL")
        self.ticker_combo.currentTextChanged.connect(self.update_stock_data)
        
        header_layout.addWidget(QLabel("Select Stock:"))
        header_layout.addWidget(self.ticker_combo)
        
        refresh_button = QPushButton("Refresh Data")
        refresh_button.clicked.connect(self.update_stock_data)
        header_layout.addWidget(refresh_button)
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Create content area with chart and data panels
        content_layout = QHBoxLayout()
        
        # Left panel for chart
        self.chart_layout = QVBoxLayout()
        self.chart_frame = QFrame()
        self.chart_frame.setFrameShape(QFrame.StyledPanel)
        self.chart_frame.setLayout(self.chart_layout)
        content_layout.addWidget(self.chart_frame, 7)  # Chart takes 70% of width
        
        # Right panel for data and predictions
        data_layout = QVBoxLayout()
        self.data_frame = QFrame()
        self.data_frame.setFrameShape(QFrame.StyledPanel)
        
        # Grid layout for data points
        self.data_grid = QGridLayout()
        self.data_frame.setLayout(self.data_grid)
        
        content_layout.addWidget(self.data_frame, 3)  # Data takes 30% of width
        main_layout.addLayout(content_layout)
        
        # Set the main widget
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize with default stock
        self.update_stock_data()
    
    def update_stock_data(self):
        ticker = self.ticker_combo.currentText()
        
        # Get stock data
        current_price, history = get_stock_data(ticker)
        
        # Calculate indicators
        rsi = calculate_rsi(history)
        predicted_price = predict_price(ticker, history)
        
        # Get multi-day predictions
        predictions = predict_price_range(ticker, history, days=5)
        
        # Update chart
        self.update_chart(ticker, history)
        
        # Update data display
        self.update_data_display(ticker, current_price, rsi, predicted_price, predictions)
    
    def update_chart(self, ticker, history):
        # Clear previous chart
        for i in reversed(range(self.chart_layout.count())): 
            self.chart_layout.itemAt(i).widget().setParent(None)
        
        # Convert the history DataFrame to format required by mplfinance
        # Make sure the index is a DatetimeIndex
        history_mpf = history.copy()
        if not isinstance(history_mpf.index, pd.DatetimeIndex):
            history_mpf.index = pd.to_datetime(history_mpf.index)
        
        # Create candlestick chart - using the correct approach for mplfinance
        fig, axes = mpf.plot(
            history_mpf, 
            type='candle', 
            style='yahoo', 
            title=f'{ticker} Stock Price', 
            ylabel='Price ($)',
            volume=True,
            figsize=(10, 6),
            returnfig=True
        )
        
        # Add the figure to the layout
        canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(canvas)
        
        # Add chart title
        chart_title = QLabel(f"{ticker} Stock Price Chart")
        chart_title.setAlignment(Qt.AlignCenter)
        chart_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.chart_layout.insertWidget(0, chart_title)
    
    def update_data_display(self, ticker, current_price, rsi, predicted_price, predictions):
        # Clear previous data
        for i in reversed(range(self.data_grid.count())): 
            item = self.data_grid.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Current date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Add title
        title = QLabel(f"{ticker} Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        self.data_grid.addWidget(title, 0, 0, 1, 2)
        
        # Current price with date
        self.data_grid.addWidget(QLabel("Current Price :"), 1, 0)
        price_label = QLabel(f"${current_price:.2f} ({today})")
        price_label.setStyleSheet("font-weight: bold;")
        self.data_grid.addWidget(price_label, 1, 1)
        
        # RSI indicator
        self.data_grid.addWidget(QLabel("RSI:"), 2, 0)
        rsi_label = QLabel(f"{rsi:.2f}")
        # Color RSI based on value
        if rsi > 70:
            rsi_label.setStyleSheet("color: red;")  # Overbought
        elif rsi < 30:
            rsi_label.setStyleSheet("color: green;")  # Oversold
        self.data_grid.addWidget(rsi_label, 2, 1)
        
        # Today's prediction
        self.data_grid.addWidget(QLabel("Today's Prediction:"), 3, 0)
        self.data_grid.addWidget(QLabel(f"${predicted_price:.2f}"), 3, 1)
        
        # Future predictions
        row = 4
        for date, price in predictions.items():
            if date != today:  # Skip today as we already displayed it
                self.data_grid.addWidget(QLabel(f"Prediction:"), row, 0)
                pred_label = QLabel(f"${price:.2f}")
                # Color based on comparison to current price
                if price > current_price:
                    pred_label.setStyleSheet("color: green;")
                elif price < current_price:
                    pred_label.setStyleSheet("color: red;")
                self.data_grid.addWidget(pred_label, row, 1)
                row += 1
        
        # Add buy/sell
        recommendation = "BUY" if predicted_price > current_price else "SELL" if predicted_price < current_price else "HOLD"
        self.data_grid.addWidget(QLabel("AI Recommendation:"), row, 0)
        rec_label = QLabel(recommendation)
        if recommendation == "BUY":
            rec_label.setStyleSheet("color: green; font-weight: bold;")
        elif recommendation == "SELL":
            rec_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            rec_label.setStyleSheet("color: orange; font-weight: bold;")
        self.data_grid.addWidget(rec_label, row, 1)
        
        # Add spacer at the bottom
        self.data_grid.setRowStretch(row + 1, 1)
