import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# --------------------- Define Stock Tickers ---------------------
# Using a predefined list to avoid Selenium/SSL issues
# You can expand this list or fetch dynamically using an API if needed
STOCK_TICKERS = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "ZOMATO.NS": "Zomato Ltd.",
    "GOOG": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "AMD": "Advanced Micro Devices Inc.",
    "NFLX": "Netflix Inc.",
    "DIS": "The Walt Disney Company",
    "BA": "The Boeing Company",
    "PYPL": "PayPal Holdings Inc.",
    "SQ": "Block Inc.",
    "SHOP": "Shopify Inc.",
    "UBER": "Uber Technologies Inc.",
    "LYFT": "Lyft Inc.",
    "INTC": "Intel Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "WFC": "Wells Fargo & Company",
    "C": "Citigroup Inc.",
    "GS": "The Goldman Sachs Group Inc.",
    "COIN": "Coinbase Global Inc.",
    "HOOD": "Robinhood Markets Inc.",
    "PFE": "Pfizer Inc.",
    "MRNA": "Moderna Inc.",
    "GME": "GameStop Corp.",
}

# --------------------- Fetch Stock Data ---------------------
def get_stock_data(ticker, period="2y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# --------------------- Model Training ---------------------
def train_linear_regression(df):
    try:
        df = df[['Close']].dropna()
        if len(df) < 10:  # Ensure enough data points
            raise ValueError("Not enough data for Linear Regression training.")
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = Ridge(alpha=0.01).fit(X_poly, y)
        return model, poly
    except Exception as e:
        st.error(f"Error training Linear Regression: {e}")
        return None, None

def train_random_forest(df):
    try:
        df = df[['Close']].dropna()
        if len(df) < 10:
            raise ValueError("Not enough data for Random Forest training.")
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error training Random Forest: {e}")
        return None

def build_lstm_model(hp):
    model = Sequential([
        LSTM(hp.Int('units', 32, 64, step=32), return_sequences=True, input_shape=(30, 1)),
        Dropout(hp.Float('dropout', 0.2, 0.4, step=0.1)),
        LSTM(hp.Int('units', 32, 64, step=32)),
        Dense(10)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

def train_lstm(df):
    try:
        df = df[['Close']].dropna()
        if len(df) < 40:  # Need at least 30 for input + 10 for output
            raise ValueError("Not enough data for LSTM training.")
        data = df.values.reshape(-1, 1)
        X, y = [], []
        for i in range(30, len(data) - 10):
            X.append(data[i-30:i])
            y.append(data[i:i+10].flatten())
        X, y = np.array(X), np.array(y)
        tuner = kt.RandomSearch(
            build_lstm_model,
            objective='val_loss',
            max_trials=1,  # Reduced for faster execution
            executions_per_trial=1,
            directory='lstm_tuner',
            project_name='stock_prediction'
        )
        tuner.search(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        return tuner.get_best_models(num_models=1)[0]
    except Exception as e:
        st.error(f"Error training LSTM: {e}")
        return None

# --------------------- Model Comparison and Ensemble ---------------------
def compare_models(df, lin_model, rf_model, lstm_model, poly):
    try:
        X_future = np.arange(len(df), len(df) + 10).reshape(-1, 1)
        X_poly_future = poly.transform(X_future)
        X_lstm_future = df['Close'].values[-30:].reshape(1, 30, 1)

        pred_lin = lin_model.predict(X_poly_future)
        pred_rf = rf_model.predict(X_future)
        pred_lstm = lstm_model.predict(X_lstm_future, verbose=0)[0]

        y_test = df['Close'].values[-10:] if len(df) >= 10 else df['Close'].values
        rmse_lin = np.sqrt(mean_squared_error(y_test, pred_lin[:len(y_test)]))
        rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf[:len(y_test)]))
        rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm[:len(y_test)]))

        return {
            'Linear Regression': rmse_lin,
            'Random Forest': rmse_rf,
            'LSTM': rmse_lstm
        }, [pred_lin, pred_rf, pred_lstm]
    except Exception as e:
        st.error(f"Error comparing models: {e}")
        return None, None

def average_predictions(pred_lin, pred_rf, pred_lstm):
    return (pred_lin + pred_rf + pred_lstm) / 3

# --------------------- Streamlit UI ---------------------
st.title("📊 Advanced Stock Prediction App")

# Initialize session state for storing input
if 'stock_input' not in st.session_state:
    st.session_state.stock_input = ""
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Create columns for input layout
col1, col2 = st.columns([2, 3])

# Function to update suggestions based on input
def update_suggestions():
    input_text = st.session_state.stock_input.upper()
    if input_text:
        # Filter suggestions that start with the input text
        primary_suggestions = [ticker for ticker in STOCK_TICKERS if ticker.startswith(input_text)]
        # Also include suggestions that contain the input text but don't start with it
        secondary_suggestions = [ticker for ticker in STOCK_TICKERS if (input_text in ticker) and not ticker.startswith(input_text)]
        # Additionally, check for company names containing the input
        company_matches = [ticker for ticker, company in STOCK_TICKERS.items() 
                          if input_text in company.upper() and ticker not in primary_suggestions + secondary_suggestions]
        # Combine all lists, prioritizing exact matches
        st.session_state.suggestions = primary_suggestions + secondary_suggestions + company_matches
    else:
        st.session_state.suggestions = list(STOCK_TICKERS.keys())[:10]  # Show top 10 by default
        
    # Make sure we always have at least some suggestions
    if not st.session_state.suggestions and input_text:
        # Try a more forgiving search
        st.session_state.suggestions = [ticker for ticker, name in STOCK_TICKERS.items() 
                                      if any(part in ticker.upper() or part in name.upper() 
                                           for part in input_text.split())]

# Initialize suggestions
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = list(STOCK_TICKERS.keys())[:10]

# Add a clear search button function
def clear_search():
    st.session_state.stock_input = ""
    update_suggestions()

# Create the text input that triggers update on change
with col1:
    st.text_input(
        "Enter Stock Ticker or Company Name",
        value=st.session_state.stock_input,
        key="stock_input",
        on_change=update_suggestions,
        placeholder="Type to search (e.g., AAPL, Tesla, Zomato)"
    )
    if st.session_state.stock_input:
        st.button("Clear Search", on_click=clear_search)

# Display suggestions with company names
with col2:
    # Show the number of matches
    if st.session_state.suggestions:
        st.caption(f"Found {len(st.session_state.suggestions)} matching stocks")
    
    # Create a more informative display of ticker options
    options = {f"{ticker} - {STOCK_TICKERS[ticker]}": ticker for ticker in st.session_state.suggestions}
    
    # Use selectbox for the dropdown experience
    if options:
        selected_option = st.selectbox(
            "Select from suggestions",
            options=list(options.keys()),
            index=0 if options else None,
            key="ticker_selector"
        )

# Add button to add the selected ticker
if 'ticker_selector' in st.session_state and st.session_state.ticker_selector:
    selected_option = st.session_state.ticker_selector
    selected_ticker = options[selected_option]
    if st.button(f"Add {selected_ticker} to Selection", key="add_ticker_button"):
        if selected_ticker not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(selected_ticker)
            st.rerun()  # Rerun to update the UI

# Display selected tickers with option to remove
st.subheader("Selected Tickers:")

if st.session_state.selected_tickers:
    # Create a container with a colored background to make selections more visible
    ticker_container = st.container()
    with ticker_container:
        # Use columns for layout
        cols = st.columns(min(len(st.session_state.selected_tickers), 3))
        
        # Create a temporary list to avoid modifying during iteration
        tickers_to_remove = []
        
        for i, ticker in enumerate(st.session_state.selected_tickers):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Create a card-like display for each ticker
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px; border-radius: 5px; background-color: #f0f2f6; border: 1px solid #ddd;">
                    <h4 style="margin:0">{ticker}</h4>
                    <p style="margin:0; font-size: 0.8em; color: #666;">{STOCK_TICKERS.get(ticker, "")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Remove {ticker}", key=f"remove_{ticker}"):
                    tickers_to_remove.append(ticker)
        
        # Remove tickers marked for deletion
        for ticker in tickers_to_remove:
            st.session_state.selected_tickers.remove(ticker)
            st.rerun()  # Rerun to update the UI immediately
            
    # Convert selected tickers to comma-separated string
    stock_tickers = ",".join(st.session_state.selected_tickers)
else:
    st.info("No tickers selected. Please search and add at least one ticker above.")
    stock_tickers = ""

model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "LSTM", "Ensemble"])

if st.button("Get Stock Data and Compare Models"):
    if not stock_tickers:
        st.error("Please select at least one stock ticker.")
        st.stop()

    # Display a loading indicator 
    with st.spinner("Fetching stock data and training models... This may take a moment."):
        tickers = [ticker.strip() for ticker in stock_tickers.split(',')]
        
        # Process tickers with better error handling
        dfs = {}
        for ticker in tickers:
            try:
                st.info(f"Fetching data for {ticker}...")
                df = get_stock_data(ticker)
                if not df.empty:
                    dfs[ticker] = df
                else:
                    st.warning(f"No data available for {ticker}")
            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
                continue

    if not any(df.shape[0] > 0 for df in dfs.values()):
        st.error("No data available for the selected tickers.")
        st.stop()

    for ticker, df in dfs.items():
        if df.empty:
            st.warning(f"No data for {ticker}. Skipping.")
            continue

        st.subheader(f"{ticker} - {STOCK_TICKERS.get(ticker, ticker)}")
        st.write(df.tail())

        # Train models
        lin_model, poly = train_linear_regression(df)
        rf_model = train_random_forest(df)
        lstm_model = train_lstm(df)

        # Check if models were trained successfully
        if lin_model is None or rf_model is None or lstm_model is None or poly is None:
            st.error("One or more models failed to train. Skipping this ticker.")
            continue

        # Compare models
        rmse_scores, predictions = compare_models(df, lin_model, rf_model, lstm_model, poly)
        if rmse_scores is None or predictions is None:
            st.error("Model comparison failed. Skipping this ticker.")
            continue

        # Plot RMSE Comparison
        rmse_df = pd.DataFrame(rmse_scores.items(), columns=['Model', 'RMSE'])
        st.bar_chart(rmse_df.set_index('Model'))

        # Plot Predictions
        future_dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Prices", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions[0], name="Linear Predictions", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions[1], name="RF Predictions", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions[2], name="LSTM Predictions", line=dict(color="red")))

        if model_choice == "Ensemble":
            ensemble_pred = average_predictions(predictions[0], predictions[1], predictions[2])
            fig.add_trace(go.Scatter(x=future_dates, y=ensemble_pred, name="Ensemble Prediction", line=dict(color="purple")))

        fig.update_layout(title=f"{ticker} Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Recommendation
        last_price = df['Close'].iloc[-1]
        predicted_price = predictions[0][-1]  # Using Linear Regression for recommendation
        if predicted_price > last_price * 1.05:
            st.success(f"Recommendation for {ticker}: BUY")
        elif predicted_price < last_price * 0.95:
            st.error(f"Recommendation for {ticker}: SELL")
        else:
            st.info(f"Recommendation for {ticker}: HOLD")