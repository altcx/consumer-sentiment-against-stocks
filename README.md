# Consumer Sentiment Against Stocks

This Python program analyzes the relationship between news sentiment and stock prices for a given stock ticker. It fetches news headlines, analyzes their sentiment, retrieves historical stock price data, and visualizes both on a graph for comparison. It's pretty reliant on how much info there is so equities will perform at various levels of accuracy - most success was found in larger groups such as PLTR. 

## Features
- Fetches news headlines for a stock ticker using Finnhub (default, recommended for most articles), NewsAPI (legacy), or generates demo data
- Analyzes sentiment of each headline using NLTK's VADER sentiment analyzer
- Fetches historical stock price data using Alpaca API (or generates demo data)
- Aggregates sentiment scores by date
- Visualizes stock price and sentiment on a dual-axis graph (stock price as a line, sentiment as color-coded bars)
- Calculates and displays the correlation between sentiment and stock price

## Requirements
- Python 3.7+
- Packages: `requests`, `pandas`, `nltk`, `matplotlib`, `numpy`, `alpaca-py`

## Setup
1. Install dependencies:
   ```powershell
   pip install requests pandas nltk matplotlib numpy alpaca-py
   ```
2. Obtain API keys:
   - [Finnhub](https://finnhub.io/): For fetching news headlines (free tier, recommended for most articles)
   - [NewsAPI](https://newsapi.org/): (optional, legacy fallback)
   - [Alpaca](https://alpaca.markets/): For fetching stock price data (free tier available)
3. Edit `sentiment_analysis.py` and replace the placeholder strings `'YOUR_FINNHUB_API_KEY'`, `'YOUR_NEWSAPI_KEY_HERE'`, and `'YOUR_ALPACA_API_KEY', 'YOUR_ALPACA_SECRET_KEY'` with your own API keys as needed.

## Usage
Run the script and enter a stock ticker symbol when prompted:
```powershell
python sentiment_analysis.py
```
- The program will fetch news and stock data for the last 30 days (using Finnhub by default).
- It will print out the headlines, their sentiment scores, and display a graph comparing sentiment and stock price.
- The graph is saved as a PNG file in the current directory.

If you do not provide API keys, the program will generate demo data for demonstration purposes.

## Example Output
- A graph showing stock price (blue line) and daily sentiment (green/red bars) for the selected stock.
- Correlation value between sentiment and stock price.
