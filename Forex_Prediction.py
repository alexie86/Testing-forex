import streamlit as st
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import load_model
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize NewsAPI
newsapi = NewsApiClient(api_key="e280eef9ec8c469fb1b0b593db736f76")

# Load pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Sidebar inputs
pair = st.sidebar.selectbox("Currency Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "ZAR=X"])
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "30m", "1d", "1mo"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1d", "1wk", "1mo", "3mo", "1y", "5y", "10y"])
model_option = st.sidebar.radio("Select Model", ["Pre-Trained", "Train New Model"])

# Function to fetch Forex data
def fetch_forex_data(pair='EURUSD=X', interval='1d', period='1y'):
    end_date = datetime.now()  # Current date
    start_date = end_date - pd.DateOffset(years=1)  # Default 1 year
    if timeframe == "1d":
        start_date = end_date - pd.DateOffset(days=1)
    elif timeframe == "1wk":
        start_date = end_date - pd.DateOffset(weeks=1)
    elif timeframe == "1mo":
        start_date = end_date - pd.DateOffset(months=1)
    elif timeframe == "3mo":
        start_date = end_date - pd.DateOffset(months=3)
    elif timeframe == "1y":
        start_date = end_date - pd.DateOffset(years=1)
    elif timeframe == "5y":
        start_date = end_date - pd.DateOffset(years=5)
    elif timeframe == "10y":
        start_date = end_date - pd.DateOffset(years=10)

    data = yf.download(pair, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
    if not data.empty:
        # Standardize column names for consistency
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return data

data = fetch_forex_data(pair=pair, interval=interval, period=timeframe)
if data.empty:
    st.error("No data fetched. Please try a different timeframe or interval.")
    st.stop()


# def fetch_forex_news(pair):
#     """
#     Fetch relevant forex news for the selected currency pair.
#     """
#     # Map currency pairs to their base currencies
#     currency_map = {
#         "EURUSD=X": ("EUR", "USD"),
#         "GBPUSD=X": ("GBP", "USD"),
#         "USDJPY=X": ("USD", "JPY"),
#         "ZAR=X": ("ZAR", "USD"),
#     }
#
#     # Identify base and quote currencies
#     base_currency, quote_currency = currency_map.get(pair, ("forex", "forex"))
#
#     # Create a more specific query
#     query = (
#         f"({base_currency} OR {quote_currency}) AND (exchange OR forex OR currency market OR "
#         f"interest rate OR monetary policy OR central bank)"
#     )
#
#     # Fetch articles using NewsAPI
#     news = newsapi.get_everything(
#         q=query,
#         language='en',
#         domains='reuters.com, bloomberg.com, wsj.com, cnbc.com, ft.com, forex.com, investing.com',
#         page_size=50
#     )
#     headlines = [article['title'] for article in news['articles']]
#
#     # Filter articles for relevance
#     filtered_headlines = filter_relevant_articles(headlines, base_currency, quote_currency)
#
#     if not filtered_headlines:
#         st.warning("No relevant news articles found. Predictions may lack sentiment adjustment.")
#
#     return pd.DataFrame({'Headline': filtered_headlines})

def fetch_forex_news(pair):
    """
    Fetch relevant forex news for the selected currency pair.
    """
    # Identify base and quote currencies from the pair
    if "=" in pair:
        base_currency, quote_currency = pair[:3], pair[3:6]
    elif "/" in pair:
        base_currency, quote_currency = pair.split("/")
    else:
        base_currency, quote_currency = pair[:3], pair[3:]

    # Construct query string
    query = (
        f"({base_currency} OR {quote_currency}) AND "
        f"(forex OR currency market OR exchange rate OR monetary policy OR central bank OR interest rate)"
    )

    try:
        # Fetch articles from NewsAPI
        news = newsapi.get_everything(
            q=query,
            language='en',
            domains='reuters.com, bloomberg.com, wsj.com, cnbc.com, ft.com, forex.com, investing.com',
            page_size=50,
            sort_by="relevancy"
        )
        headlines = [article['title'] for article in news['articles']]

        # Filter articles for relevance
        filtered_headlines = filter_relevant_articles(headlines, base_currency, quote_currency)

        if not filtered_headlines:
            st.warning("No relevant news articles found. Predictions may lack sentiment adjustment.")

        return pd.DataFrame({'Headline': filtered_headlines})

    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return pd.DataFrame({'Headline': []})

# Preprocess and vectorize headlines
def preprocess_headlines(headlines):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(headlines)
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Topic modeling using LDA
def topic_modeling(tfidf_matrix, n_topics=3):
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(tfidf_matrix)
    return lda_model


# def filter_relevant_articles(headlines, base_currency, quote_currency):
#     """
#     Filter news headlines based on relevance to the forex market and currency pair.
#     """
#     # Define relevance keywords
#     relevant_keywords = [
#         "forex", "currency market", "exchange rate", "monetary policy",
#         "central bank", "interest rate", base_currency, quote_currency
#     ]
#
#     # Filter headlines containing at least one keyword
#     relevant_headlines = [
#         headline for headline in headlines
#         if any(keyword.lower() in headline.lower() for keyword in relevant_keywords)
#     ]
#     return relevant_headlines

def filter_relevant_articles(headlines, base_currency, quote_currency):
    """
    Filter news headlines based on relevance to the forex market and currency pair.
    """
    # Define relevance keywords
    relevant_keywords = [
        "forex", "currency market", "exchange rate", "monetary policy",
        "central bank", "interest rate", base_currency, quote_currency
    ]

    # Filter headlines containing at least one keyword
    relevant_headlines = [
        headline for headline in headlines
        if any(keyword.lower() in headline.lower() for keyword in relevant_keywords)
    ]
    return relevant_headlines

# Function to analyze sentiment of news headlines
def analyze_sentiment(news_df):
    news_df['Sentiment'] = news_df['Headline'].apply(
        lambda headline: sentiment_model(headline)[0]['label']
    )
    return news_df

# Function to add technical indicators
def add_technical_indicators(data):
    if len(data) < 14:
        raise ValueError("Not enough data to calculate RSI. Ensure at least 14 rows of data.")

    # Calculate SMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate Volatility
    data['Volatility'] = data['Close'].rolling(window=20).std()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Handle missing values by forward-filling or backfilling
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)

    data['RSI'].fillna(method='bfill', inplace=True)
    data['RSI'].fillna(method='ffill', inplace=True)

    return data

# Fetch and preprocess data
data = fetch_forex_data(pair=pair, interval=interval, period=timeframe)
data = add_technical_indicators(data)

def add_features(data):
    # Moving Averages
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # MACD and Signal Line
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    data['Upper_Band'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)

    # Lagged Features
    data['Close_Lag_1'] = data['Close'].shift(1)
    data['Close_Lag_2'] = data['Close'].shift(2)

    # Rolling Max/Min
    data['Rolling_Max_20'] = data['Close'].rolling(window=20).max()
    data['Rolling_Min_20'] = data['Close'].rolling(window=20).min()

    # Fill missing values
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    return data

# Add features to your dataset
data = add_features(data)

# Function to prepare data for LSTM model
def prepare_data(data, seq_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_20', 'SMA_50', 'RSI']])

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length, 0])
    return np.array(X), np.array(y), scaler

# Function to build LSTM model
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit app setup
st.title("Forex Prediction System")

# Fetch and preprocess data
data = fetch_forex_data(pair=pair, interval=interval, period=timeframe)
data = add_technical_indicators(data)

# Fetch Forex news and analyze sentiment
news_data = fetch_forex_news(pair)
news_data = analyze_sentiment(news_data)
# Fetch Forex news and analyze sentiment
# news_data = fetch_forex_news(pair)
if not news_data.empty:
    news_data['Sentiment'] = news_data['Headline'].apply(
        lambda headline: sentiment_model(headline)[0]['label']
    )
    st.subheader("Filtered Forex News Sentiment")
    st.write(news_data)

# Fetch Forex news and analyze sentiment
# news_data = fetch_forex_news(pair)
# if not news_data.empty:
#     news_data = analyze_sentiment(news_data)
#     st.subheader("Filtered Forex News Sentiment")
#     st.write(news_data)

# Display news sentiment
st.subheader("Recent Forex News Sentiment")
st.write(news_data)
positive_sentiment = news_data['Sentiment'].value_counts().get('POSITIVE', 0)
negative_sentiment = news_data['Sentiment'].value_counts().get('NEGATIVE', 0)

st.write(f"Positive Sentiment: {positive_sentiment}")
st.write(f"Negative Sentiment: {negative_sentiment}")

# Prepare data for model
seq_length = 60
X, y, scaler = prepare_data(data, seq_length)

# Prepare data for LSTM model
seq_length = 60
if len(data) > seq_length:  # Ensure there is enough data
    X, y, scaler = prepare_data(data, seq_length)
else:
    st.error(
        "Not enough data points to prepare LSTM sequences. Please adjust lookback period or check the data source.")
    st.stop()

# Train new model or load pre-trained model
if model_option == "Pre-Trained":
    try:
        model = load_model("models/forex_model.h5")
        st.write("Loaded pre-trained model.")
    except Exception as e:
        st.error(f"Error loading pre-trained model: {e}")
        st.stop()
else:
    st.write("Training a new LSTM model...")
    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm(input_shape)

    # Train the model
    try:
        model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
        # Save model with a dynamic filename
        model_filename = f"models/forex_model_{time.strftime('%Y%m%d%H%M%S')}.h5"
        model.save(model_filename)
        st.write(f"New model trained and saved as '{model_filename}'")
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.stop()

# Make prediction
prediction = model.predict(X[-1].reshape(1, -1, X.shape[2]))
predicted_price = scaler.inverse_transform(
    np.concatenate([prediction, np.zeros((1, X.shape[2] - 1))], axis=1)
)[0, 0]

# Adjust prediction based on sentiment
if positive_sentiment > negative_sentiment:
    sentiment_factor = 1.01
    st.write("Market sentiment is positive. Adjusting prediction upwards.")
elif negative_sentiment > positive_sentiment:
    sentiment_factor = 0.99
    st.write("Market sentiment is negative. Adjusting prediction downwards.")
else:
    sentiment_factor = 1.00
    st.write("Market sentiment is neutral. No adjustment to prediction.")

adjusted_price = predicted_price * sentiment_factor

st.subheader(f"Predicted Next Price for {pair}: {adjusted_price}")

import plotly.graph_objects as go

data = add_technical_indicators(data)
# Calculate the scaled RSI
rsi_min, rsi_max = 0, 100  # RSI is always between 0 and 100
price_min = data['Close'].min()  # Minimum value in the Close prices
price_max = data['Close'].max()  # Maximum value in the Close prices

# Scale RSI to match the Close price range
data['RSI_Scaled'] = (data['RSI'] - rsi_min) * (price_max - price_min) / (rsi_max - rsi_min) + price_min

# Create the main figure
fig_combined = go.Figure()

# Add Close price to the chart
fig_combined.add_trace(go.Scatter(
    x=data.index, y=data['Close'], mode='lines', name='Close Price',
    line=dict(color='blue', width=2)
))

# Add SMA_20 to the chart
fig_combined.add_trace(go.Scatter(
    x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20',
    line=dict(color='orange', width=2, dash='dash')
))

# Add SMA_50 to the chart
fig_combined.add_trace(go.Scatter(
    x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50',
    line=dict(color='green', width=2, dash='dot')
))

# Add Scaled RSI line to the chart
fig_combined.add_trace(go.Scatter(
    x=data.index, y=data['RSI_Scaled'], mode='lines', name='RSI (Scaled)',
    line=dict(color='purple', width=2, dash='dashdot')
))

# Add overbought line (scaled to price range)
rsi_overbought_scaled = (70 - rsi_min) * (price_max - price_min) / (rsi_max - rsi_min) + price_min
fig_combined.add_shape(
    type="line", x0=data.index[0], x1=data.index[-1], y0=rsi_overbought_scaled, y1=rsi_overbought_scaled,
    line=dict(color="red", width=1, dash="dash"),
    name="RSI Overbought"
)

# Add oversold line (scaled to price range)
rsi_oversold_scaled = (30 - rsi_min) * (price_max - price_min) / (rsi_max - rsi_min) + price_min
fig_combined.add_shape(
    type="line", x0=data.index[0], x1=data.index[-1], y0=rsi_oversold_scaled, y1=rsi_oversold_scaled,
    line=dict(color="green", width=1, dash="dash"),
    name="RSI Oversold"
)

# Customize the layout
fig_combined.update_layout(
    title="Close Price, SMA, and RSI Overbought/Oversold Levels",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

# Display the combined chart
st.subheader("Price and RSI Levels Chart (Scaled)")
st.plotly_chart(fig_combined)

# Explain each indicator
st.write("""
- **SMA_20**: Simple Moving Average of the last 20 periods.
- **SMA_50**: Simple Moving Average of the last 50 periods.
- **Volatility**: Standard deviation of the closing prices over the last 20 periods.
- **RSI**: Relative Strength Index, calculated over the last 14 periods.
""")

# Add checkboxes for chart control
st.subheader("Customize Chart Display")
show_close = st.checkbox("Show Close Price", value=True)
show_sma_20 = st.checkbox("Show SMA 20", value=True)
show_sma_50 = st.checkbox("Show SMA 50", value=True)
show_rsi = st.checkbox("Show RSI (Scaled)", value=True)

# Create the main figure
fig_combined = go.Figure()

# Add selected traces to the chart
if show_close:
    fig_combined.add_trace(go.Scatter(
        x=data.index, y=data['Close'], mode='lines', name='Close Price',
        line=dict(color='blue', width=2)
    ))

if show_sma_20:
    fig_combined.add_trace(go.Scatter(
        x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20',
        line=dict(color='orange', width=2, dash='dash')
    ))

if show_sma_50:
    fig_combined.add_trace(go.Scatter(
        x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50',
        line=dict(color='green', width=2, dash='dot')
    ))

if show_rsi:
    # Add RSI line to the chart
    fig_combined.add_trace(go.Scatter(
        x=data.index, y=data['RSI_Scaled'], mode='lines', name='RSI (Scaled)',
        line=dict(color='purple', width=2, dash='dashdot')
    ))

    # Add overbought and oversold levels
    rsi_overbought_scaled = (70 - rsi_min) * (price_max - price_min) / (rsi_max - rsi_min) + price_min
    fig_combined.add_shape(
        type="line", x0=data.index[0], x1=data.index[-1], y0=rsi_overbought_scaled, y1=rsi_overbought_scaled,
        line=dict(color="red", width=1, dash="dash"),
        name="RSI Overbought"
    )

    rsi_oversold_scaled = (30 - rsi_min) * (price_max - price_min) / (rsi_max - rsi_min) + price_min
    fig_combined.add_shape(
        type="line", x0=data.index[0], x1=data.index[-1], y0=rsi_oversold_scaled, y1=rsi_oversold_scaled,
        line=dict(color="green", width=1, dash="dash"),
        name="RSI Oversold"
    )

# Customize the layout
fig_combined.update_layout(
    title="Customized Chart Display",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

# Display the combined chart
st.plotly_chart(fig_combined)

# def add_features(data):
#     # Exponential Moving Average
#     data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
#
#     # MACD and Signal Line
#     data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
#     data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
#
#     # Bollinger Bands
#     data['SMA_20'] = data['Close'].rolling(window=20).mean()
#     data['Upper_Band'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
#     data['Lower_Band'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
#
#     # Fill missing values
#     data.fillna(method='bfill', inplace=True)
#     data.fillna(method='ffill', inplace=True)
#     return data

import plotly.graph_objects as go

# Ensure features are added to the data
data = add_features(data)

# Define a function to create interactive charts with selectable features
def plot_chart(data, selected_features):
    fig = go.Figure()

    # Add features to the chart based on user selection
    if 'Close' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Close Price',
            line=dict(color='blue', width=2)
        ))
    if 'SMA_20' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20',
            line=dict(color='orange', width=2, dash='dash')
        ))
    if 'SMA_50' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50',
            line=dict(color='green', width=2, dash='dot')
        ))
    if 'RSI' in selected_features:
        rsi_min, rsi_max = 0, 100  # RSI is always between 0 and 100
        price_min, price_max = data['Close'].min(), data['Close'].max()
        data['RSI_Scaled'] = (data['RSI'] - rsi_min) * (price_max - price_min) / (rsi_max - rsi_min) + price_min
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI_Scaled'], mode='lines', name='RSI (Scaled)',
            line=dict(color='purple', width=2, dash='dashdot')
        ))
    if 'EMA_20' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20',
            line=dict(color='cyan', width=2)
        ))
    if 'MACD' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'], mode='lines', name='MACD',
            line=dict(color='magenta', width=2)
        ))
    if 'Signal_Line' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line',
            line=dict(color='brown', width=2, dash='dot')
        ))
    if 'Upper_Band' in selected_features and 'Lower_Band' in selected_features:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Upper_Band'], mode='lines', name='Upper Bollinger Band',
            line=dict(color='gray', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Lower_Band'], mode='lines', name='Lower Bollinger Band',
            line=dict(color='gray', width=1)
        ))

    # Customize the layout
    fig.update_layout(
        title="Forex Indicators Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=600
    )
    return fig

# Sidebar checkboxes for feature selection
selected_features = st.sidebar.multiselect(
    "Select Features to Display on the Chart:",
    ['Close', 'SMA_20', 'SMA_50', 'RSI', 'EMA_20', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band'],
    default=['Close', 'SMA_20', 'SMA_50', 'RSI']  # Default features to show
)

# Generate and display the chart
st.subheader("Forex Indicators Chart")
chart = plot_chart(data, selected_features)
st.plotly_chart(chart)

# Explain each indicator
st.write("""
### Indicator Explanations:
- **Close**: The closing price of the Forex pair.
- **SMA_20 & SMA_50**: Simple Moving Averages calculated over 20 and 50 periods, respectively.
- **RSI**: Relative Strength Index, scaled to match the price range.
- **EMA_20**: Exponential Moving Average over 20 periods.
- **MACD & Signal Line**: MACD (12, 26) and its Signal Line (9).
- **Upper_Band & Lower_Band**: Bollinger Bands (2 standard deviations around the SMA_20).
""")