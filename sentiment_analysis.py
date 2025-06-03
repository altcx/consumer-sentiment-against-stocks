import requests
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import random
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')


def fetch_news_headlines(stock_ticker, start_date, end_date):
    NEWSAPI_KEY = '150417af534544cc93a738e004b99d9b'
    url = 'https://newsapi.org/v2/everything'
    headlines_with_dates = []
    # NewsAPI allows up to 100 results per page, so we can paginate for more articles ()
    page = 1
    max_pages = 5  # Try to get up to 500 articles (within free tier limits -> I'm broke)
    while page <= max_pages:
        params = {
            'q': stock_ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWSAPI_KEY,
            'pageSize': 100,
            'page': page
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            for article in articles:
                headline = article['title']
                pub_date = article.get('publishedAt', '')[:10]
                headlines_with_dates.append((headline, pub_date))
            if len(articles) < 100:
                break  # No more pages
        else:
            print(f"Failed to fetch news: {response.status_code}")
            break
        page += 1
    return headlines_with_dates


def fetch_news_headlines_finnhub(stock_ticker, start_date, end_date):
    FINNHUB_API_KEY = 'YOUR_FINNHUB_API_KEY'  # <-- Insert your Finnhub API key here
    url = 'https://finnhub.io/api/v1/company-news'
    params = {
        'symbol': stock_ticker,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }
    response = requests.get(url, params=params)
    headlines_with_dates = []
    if response.status_code == 200:
        articles = response.json()
        for article in articles:
            headline = article.get('headline', '')
            pub_date = datetime.utcfromtimestamp(article['datetime']).strftime('%Y-%m-%d')
            headlines_with_dates.append((headline, pub_date))
    else:
        print(f"Finnhub news fetch failed: {response.status_code}")
    return headlines_with_dates


def analyze_sentiment(headlines):
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(headline)['compound'] for headline in headlines]


def fetch_stock_data(stock_ticker, start_date, end_date, alpaca_client):
    request_params = StockBarsRequest(
        symbol_or_symbols=stock_ticker,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    bars = alpaca_client.get_stock_bars(request_params)
    df = pd.DataFrame([
        {'timestamp': bar.timestamp, 'close': bar.close}
        for bar in bars[stock_ticker]
    ])
    return df


def generate_demo_data(stock_ticker, start_date, end_date):
    random.seed(0)
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    headlines_with_dates = []

    very_positive_headlines = [
        f"{stock_ticker} crushes earnings expectations by 50%, stock soars!",
        f"Revolutionary product breakthrough for {stock_ticker} will reshape industry",
        f"{stock_ticker} secures massive government contract worth billions",
        f"Analysts predict {stock_ticker} could double in value this year",
        f"{stock_ticker}'s CEO wins leadership award, company culture thriving"
    ]
    moderately_positive_headlines = [
        f"{stock_ticker} reports solid earnings, slightly above expectations",
        f"New partnership announced for {stock_ticker}, showing promise",
        f"{stock_ticker} expands into emerging markets with cautious optimism",
        f"Analysts upgrade {stock_ticker} from 'hold' to 'buy' rating",
        f"{stock_ticker}'s efficiency initiatives starting to show results"
    ]
    slightly_positive_headlines = [
        f"{stock_ticker} meets earnings expectations, outlook stable",
        f"Minor product improvements announced by {stock_ticker}",
        f"Small acquisition completed by {stock_ticker}, integration underway",
        f"{stock_ticker} maintains market position despite challenges",
        f"Analysts see {stock_ticker} as resilient in current conditions"
    ]
    neutral_headlines = [
        f"{stock_ticker} announces routine management changes",
        f"Annual shareholder meeting scheduled for {stock_ticker}",
        f"{stock_ticker} to present at upcoming industry conference",
        f"No significant updates from {stock_ticker} this quarter",
        f"{stock_ticker} maintains previous guidance for fiscal year"
    ]
    slightly_negative_headlines = [
        f"{stock_ticker} slightly misses revenue targets but profits stable",
        f"Minor setback for {stock_ticker}'s expansion plans",
        f"Some analysts express concern about {stock_ticker}'s growth pace",
        f"{stock_ticker} faces increased competition in core market",
        f"Regulatory review underway for {stock_ticker}'s new venture"
    ]
    moderately_negative_headlines = [
        f"{stock_ticker} misses earnings targets, lowers guidance",
        f"Product launch delayed for {stock_ticker} due to technical issues",
        f"Market share declining for {stock_ticker} in key segment",
        f"Analysts downgrade {stock_ticker} citing industrywide pressures",
        f"{stock_ticker} announces restructuring amid profitability concerns"
    ]
    very_negative_headlines = [
        f"{stock_ticker} reports catastrophic losses, stock plummets!",
        f"Major product recall announced by {stock_ticker}, lawsuits expected",
        f"CEO of {stock_ticker} resigns amid accounting scandal investigation",
        f"Massive layoffs at {stock_ticker} as company struggles to survive",
        f"{stock_ticker} loses critical patent case, faces massive penalty"
    ]
    headline_categories = [
        (very_positive_headlines, 0.85, 1.0),
        (moderately_positive_headlines, 0.5, 0.84),
        (slightly_positive_headlines, 0.2, 0.49),
        (neutral_headlines, -0.19, 0.19),
        (slightly_negative_headlines, -0.49, -0.2),
        (moderately_negative_headlines, -0.84, -0.5),
        (very_negative_headlines, -1.0, -0.85)
    ]

    for i, date in enumerate(dates):
        weekly_cycle = np.sin(i % 7 / 3.5) * 0.3
        market_trend = np.sin(i/15) * 0.5
        random_events = random.uniform(-0.5, 0.5)
        base_sentiment = market_trend + 0.5 * weekly_cycle + 0.2 * random_events
        base_sentiment = max(min(base_sentiment, 1.0), -1.0)
        for headlines, min_score, max_score in headline_categories:
            if min_score <= base_sentiment <= max_score:
                headline = random.choice(headlines)
                sentiment_score = random.uniform(min_score, max_score)
                break
        headlines_with_dates.append((headline, date.strftime('%Y-%m-%d')))

    np.random.seed(0)
    price_changes = np.random.randn(num_days)
    price_changes = price_changes - np.mean(price_changes)
    price_changes = price_changes / np.std(price_changes)
    price_changes = price_changes * 2
    price_changes = np.cumsum(price_changes)
    base_price = 100
    stock_prices = base_price + price_changes
    stock_data = pd.DataFrame({
        'timestamp': dates,
        'close': stock_prices
    })
    return headlines_with_dates, stock_data


def process_headlines_with_sentiment(headlines_with_dates):
    sia = SentimentIntensityAnalyzer()
    scored_headlines = []
    daily_sentiment = {}
    for headline, date in headlines_with_dates:
        # Use VADER for all headlines, but boost/penalize for strong words
        score = sia.polarity_scores(headline)['compound']
        headline_lower = headline.lower()
        # Boost or penalize based on strong positive/negative words
        strong_positive = ["crushes", "soars", "breakthrough", "surge", "record", "skyrockets", "outperforms", "beats", "explodes", "rally", "booming", "bullish", "all-time high", "massive gain", "best ever", "unprecedented"]
        strong_negative = ["plummets", "catastrophic", "scandal", "lawsuit", "crash", "tumbles", "drops", "misses", "recall", "resigns", "layoffs", "bearish", "collapse", "worst ever", "fraud", "investigation", "penalty", "loss"]
        for word in strong_positive:
            if word in headline_lower:
                score += 0.25
        for word in strong_negative:
            if word in headline_lower:
                score -= 0.25
        score = max(min(score, 1.0), -1.0)
        scored_headlines.append((headline, score, date))
        if date not in daily_sentiment:
            daily_sentiment[date] = []
        daily_sentiment[date].append(score)
    daily_avg = {date: sum(scores)/len(scores) for date, scores in daily_sentiment.items()}
    daily_sentiment_df = pd.DataFrame(list(daily_avg.items()), columns=['date', 'sentiment'])
    return scored_headlines, daily_sentiment_df


def create_sentiment_stock_graph(sentiment_df, stock_df, ticker, scored_headlines=None):
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['timestamp']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    merged_data = pd.merge(sentiment_df, stock_df, on='date', how='inner')
    if merged_data.empty:
        print("No matching dates between sentiment and stock data for visualization.")
        return None
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)', color='tab:blue')
    ax1.plot(merged_data['date'], merged_data['close'], 'b-', label='Stock Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment Score', color='darkgreen')
    # Remove per-article dots/lines for clarity, keep only daily average bars
    bar_width = 0.3
    colors = ['green' if s >= 0 else 'red' for s in merged_data['sentiment']]
    bars = ax2.bar(merged_data['date'], merged_data['sentiment'], color=colors, alpha=0.7, width=bar_width, label='Daily Avg Sentiment', align='center')
    sentiment_max = max(abs(merged_data['sentiment'].max()), abs(merged_data['sentiment'].min()))
    y_padding = 0.2
    ax2.set_ylim(-sentiment_max - y_padding, sentiment_max + y_padding)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax1.grid(True, alpha=0.3)
    plt.title(f'Stock Price vs News Sentiment for {ticker} - 30-Day Analysis')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    fig.autofmt_xdate()
    correlation = merged_data['sentiment'].corr(merged_data['close'])
    textstr = f'Correlation: {correlation:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.tight_layout()
    return fig


def main():
    stock_ticker = input('Enter stock ticker (e.g., AAPL): ').strip().upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    print(f"Fetching news headlines for {stock_ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    headlines_with_dates = fetch_news_headlines_finnhub(stock_ticker, start_date, end_date)
    if not headlines_with_dates:
        print("No news headlines found or API failed. Using demo data instead.")
        headlines_with_dates, stock_data = generate_demo_data(stock_ticker, start_date, end_date)
    else:
        print(f"Fetched {len(headlines_with_dates)} headlines for {stock_ticker} (Finnhub).")
        ALPACA_API_KEY = 'YOUR_ALPACA_API_KEY'  # <-- Insert your Alpaca API key here
        ALPACA_SECRET_KEY = 'YOUR_ALPACA_SECRET_KEY'  # <-- Insert your Alpaca secret key here
        alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        stock_data = fetch_stock_data(stock_ticker, start_date, end_date, alpaca_client)
    scored_headlines, daily_sentiment = process_headlines_with_sentiment(headlines_with_dates)
    print("\nHeadlines and their sentiment scores:")
    for headline, score, date in scored_headlines:
        sentiment_marker = "+" if score > 0 else "-" if score < 0 else " "
        print(f"{date} | {sentiment_marker}{score:.4f} | {headline}")
    all_scores = [score for _, score, _ in scored_headlines]
    avg_score = sum(all_scores)/len(all_scores) if all_scores else 0
    print(f"\nOverall average sentiment score: {avg_score:.4f}")
    if not daily_sentiment.empty and not stock_data.empty:
        print("\nCreating visualization comparing sentiment and stock prices...")
        try:
            fig = create_sentiment_stock_graph(daily_sentiment, stock_data, stock_ticker, scored_headlines=scored_headlines)
            if fig:
                today_str = datetime.now().strftime("%Y%m%d")
                output_file = f"{stock_ticker}_sentiment_price_30day_{today_str}.png"
                fig.savefig(output_file)
                print(f"Visualization saved to {output_file}")
                plt.show()
                daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
                stock_data['date'] = pd.to_datetime(stock_data['timestamp']).dt.date
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                merged_data = pd.merge(daily_sentiment, stock_data, on='date', how='inner')
                if not merged_data.empty:
                    correlation = merged_data['sentiment'].corr(merged_data['close'])
                    print(f"Correlation between sentiment and stock price: {correlation:.4f}")
                    print("Days with higher sentiment tend to " + ("have higher" if correlation > 0 else "have lower") + " stock prices.")
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    else:
        print("Insufficient data for visualization.")


if __name__ == '__main__':
    main()
