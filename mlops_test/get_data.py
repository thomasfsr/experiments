import yfinance as yf
import matplotlib.pyplot as plt

# Specify the ticker symbol for the stock you're interested in
# Example: Petrobras (PBR) listed in B3
ticker_symbol = "PETR4.SA"

# Download historical market data
data = yf.download(ticker_symbol, start="2023-01-01", end="2023-12-31")

# Display the data
plt.plot(data)
