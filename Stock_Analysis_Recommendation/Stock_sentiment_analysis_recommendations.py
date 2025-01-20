from datetime import datetime, timedelta

import gradio as gr
import ollama
import yfinance as yf
from newsapi import NewsApiClient
from yahooquery import search

NEWS_API_KEY = "XXXXXXXXXXX" # API key from https://newsapi.org


def get_market_data(ticker: str, period="1mo", interval="1d"):
    """
    Retrieve market data using yfinance
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False, group_by="column")
    return data


def get_news_headlines(company_name: str, from_date: str, to_date: str, language="en", page_size=10):
    """
    Retrieve news headlines from NewsAPI.org
    """
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    response = newsapi.get_everything(q=company_name,
                                      from_param=from_date,
                                      language=language,
                                      sort_by='relevancy',
                                      page=2)
    print(f"New response: {response}")
    if response['totalResults'] > 0:
        articles = response.get("articles", [])
        headlines = [article["title"] for article in articles]
        print(f"headlines: \n {headlines}")
        return headlines
    else:
        return []


def analyze_sentiment_with_gpt(text_list):
    """
    Use OpenAI GPT to classify or summarize sentiment of a list of text strings.
    We prompt GPT with a custom instruction for each text snippet.
    """
    sentiment_results = []
    for text in text_list:
        prompt = f"""
        You are a financial sentiment analyst. 
        Determine if the following headline has a Positive, Negative, or Neutral sentiment towards the company's stock performance. 
        Headline: "{text}"
        Response (one word: Positive/Negative/Neutral):
        """
        try:
            response = ollama.chat(model="llama3.2", format="json",
                                   messages=[{"role": "user", "content": prompt}])
            sentiment = response.message.content
            sentiment_results.append((text, sentiment))
        except Exception as e:
            sentiment_results.append((text, "Error"))
    return sentiment_results


def generate_recommendation(ticker: str, sentiments: list, market_data):
    """
    Combine sentiment + market data into a final GPT-based recommendation.
    The prompt can incorporate technical or fundamental signals.
    """
    # Just a simple approach: create a large prompt with key points
    average_sentiment = "Neutral"
    if sentiments:
        # naive measure: count positives vs negatives
        positives = sum(1 for (_, s) in sentiments if s.lower() == "positive")
        negatives = sum(1 for (_, s) in sentiments if s.lower() == "negative")
        if positives > negatives:
            average_sentiment = "Positive"
        elif negatives > positives:
            average_sentiment = "Negative"
        else:
            average_sentiment = "Neutral"
    print(f"market_data:\n {market_data}")
    # Summarize recent price info:
    recent_close = market_data["Close"].iloc[-1]
    first_close = market_data["Close"].iloc[0]
    price_change = recent_close - first_close
    price_change_percent = (price_change / first_close) * 100

    prompt = f"""
    You are a financial expert AI. 
    We have the following data for ticker {ticker}:
    - Average headline sentiment: {average_sentiment}
    - Price started at {round(first_close, 2)} and ended at {round(recent_close, 2)} 
      in the given period, a change of {round(price_change_percent, 2)}%.
    - Sentiments: {[(headline, s) for (headline, s) in sentiments]}

    Based on this data, write a concise summary of the current sentiment and potential outlook 
    for the stock. Then provide a short suggestion (e.g. "buy", "hold", "sell") with reasoning. 
    Remember: This is NOT real financial advice, just a hypothetical model output.
    """

    try:
        response = ollama.chat(model="llama3.2", format="json",
                               messages=[{"role": "user", "content": prompt}])
        print(f"response: {response}")
        recommendation = response.message.content
        return recommendation
    except Exception as e:
        return f"Error generating recommendation: {e}"


def get_ticker_from_company_name(company_name: str):
    """
    Searches Yahoo Finance for the given company_name and returns
    the first EQUITY symbol if found. Otherwise returns None.
    """
    results = search(company_name)
    # `results` is typically a dictionary with keys like "quotes", "news", etc.

    # Extract the list of quotes from the result
    quotes = results.get("quotes", [])
    if not quotes:
        return None  # No results found

    # For demonstration, we pick the first item that is an EQUITY
    for q in quotes:
        if q.get("quoteType") == "EQUITY":
            # q might have keys like symbol, shortname, longname, etc.
            return q.get("symbol")  # e.g. "AAPL"

    return None  # If no EQUITY symbols found in the search


# Consolidated function for Gradio
def analyze_stock(ticker: str):
    """
    Main pipeline: fetch data, fetch headlines, analyze sentiment, generate recommendation.
    Returns two strings: (sentiment_summary, final_recommendation).
    """
    today = datetime.utcnow().date()
    one_week_ago = today - timedelta(days=7)
    look_up_ticker = get_ticker_from_company_name(ticker)
    if look_up_ticker:
        print(f"look_up_ticker: {look_up_ticker}")
        ticker = look_up_ticker
    # 1) Market data
    market_data = get_market_data(ticker, period="1mo", interval="1d")
    # 2) News headlines
    news_headlines = get_news_headlines(
        company_name=ticker,  # or "Apple" if ticker is "AAPL"
        from_date=one_week_ago.isoformat(),
        to_date=today.isoformat(),
        language="en"
    )
    # 3) Sentiment analysis
    sentiments = analyze_sentiment_with_gpt(news_headlines)
    # 4) Final recommendation
    final_recommendation = generate_recommendation(ticker, sentiments, market_data)

    # Create a nice summary of the sentiment results
    # e.g. "Headline -> Sentiment"
    sentiment_summary = "\n".join(f"{i+1}. {h} -> {s}" for i, (h, s) in enumerate(sentiments))

    return sentiment_summary, final_recommendation

def run_analysis(ticker):
    # Call our pipeline
    sentiments, recommendation = analyze_stock(ticker)
    return sentiments, recommendation

with gr.Blocks() as demo:
    gr.Markdown("# Stock Sentiment & Recommendation App")
    gr.Markdown("Enter a stock ticker (e.g. `AAPL`, `TSLA`, etc.) and click **Run**.")

    ticker_input = gr.Textbox(label="Stock Ticker", value="AAPL")
    run_button = gr.Button("Run Analysis")

    sentiment_output = gr.Textbox(label="Headline Sentiment Analysis", lines=10)
    recommendation_output = gr.Textbox(label="Final Recommendation", lines=10)

    # When run_button is clicked, call run_analysis(ticker_input), show results in the 2 outputs
    run_button.click(
        fn=run_analysis,
        inputs=[ticker_input],
        outputs=[sentiment_output, recommendation_output]
    )

demo.launch()
