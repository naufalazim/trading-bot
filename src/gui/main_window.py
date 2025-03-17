from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel  # Added QLabel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from src.data.stock_data import get_stock_data
from src.data.analysis import calculate_rsi, predict_price
from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Trading AI Bot")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Stock data 
        ticker = "AAPL"
        current_price, history = get_stock_data(ticker)
        rsi = calculate_rsi(history)
        predicted_price = predict_price(ticker, history)

        # Plot
        fig, ax = plt.subplots()
        history["Close"].plot(ax=ax)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Display data
        layout.addWidget(QLabel(f"Current Price ({today}): ${current_price:.2f}"))
        layout.addWidget(QLabel(f"Predicted Price: ${predicted_price:.2f}"))
        layout.addWidget(QLabel(f"Predicted Price: ${predicted_price:.2f}"))
        layout.addWidget(QLabel(f"Predicted Price: ${predicted_price:.2f}"))
        layout.addWidget(QLabel(f"RSI: {rsi:.2f}"))
        layout.addWidget(QLabel(f"News Sentiment: Positive"))