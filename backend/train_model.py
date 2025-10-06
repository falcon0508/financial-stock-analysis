import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate, Flatten
import joblib
import os

# -------------------------
# 1️⃣ Configuration
# -------------------------
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "JNJ",
    "PG", "MA", "UNH", "HD", "PYPL",
    "DIS", "VZ", "INTC", "CSCO", "IBM",
    "KO", "PEP", "MRK", "PFE", "ABT",
    "T", "XOM", "CVX", "BA", "GE",
    "WMT", "MCD", "NKE", "ADBE", "NFLX",
    "ORCL", "CRM", "AMD", "QCOM", "TXN",
    "TMO", "MDT", "BMY", "GILD", "AMGN",
    "LLY", "MRNA", "REGN", "VRTX", "UPS"
]

sequence_length = 60
scaler_folder = "scalers"
os.makedirs(scaler_folder, exist_ok=True)

all_X_seq, all_X_ticker, all_y = [], [], []

# -------------------------
# 2️⃣ Encode tickers as integers for embedding
# -------------------------
ticker_encoder = LabelEncoder()
ticker_encoder.fit(tickers)
num_tickers = len(ticker_encoder.classes_)
joblib.dump(ticker_encoder, "ticker_encoder.pkl")  # save for later

# -------------------------
# 3️⃣ Load & process each stock
# -------------------------
for ticker in tickers:
    print(f"Processing {ticker}...")
    data = yf.download(ticker, start="2018-01-01", end="2025-01-01")
    close_prices = data["Close"].values.reshape(-1, 1)

    if len(close_prices) < sequence_length:
        print(f"Not enough data for {ticker}, skipping...")
        continue

    # Scale prices per stock
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    joblib.dump(scaler, os.path.join(scaler_folder, f"scaler_{ticker}.pkl"))

    # Prepare sequences
    for i in range(sequence_length, len(scaled_data)):
        seq = scaled_data[i-sequence_length:i, 0]
        all_X_seq.append(seq)
        all_X_ticker.append(ticker_encoder.transform([ticker])[0])
        all_y.append(scaled_data[i, 0])

# -------------------------
# 4️⃣ Convert to numpy arrays
# -------------------------
X_seq = np.array(all_X_seq)
X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))
X_ticker = np.array(all_X_ticker)
y = np.array(all_y)
print("Sequences shape:", X_seq.shape, "Tickers shape:", X_ticker.shape, "y shape:", y.shape)

# -------------------------
# 5️⃣ Build LSTM model with ticker embedding
# -------------------------
# Stock sequence input
seq_input = Input(shape=(sequence_length, 1))

lstm_out = LSTM(50, return_sequences=True)(seq_input)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = LSTM(50, return_sequences=False)(lstm_out)
lstm_out = Dropout(0.2)(lstm_out)

# Stock embedding input
ticker_input = Input(shape=(1,))
embed = Embedding(input_dim=num_tickers, output_dim=5)(ticker_input)
embed = Flatten()(embed)

# Combine LSTM output + ticker embedding
combined = Concatenate()([lstm_out, embed])
dense = Dense(25, activation="relu")(combined)
output = Dense(1)(dense)

model = Model(inputs=[seq_input, ticker_input], outputs=output)
model.compile(optimizer="adam", loss="mean_squared_error")

# -------------------------
# 6️⃣ Train model
# -------------------------
model.fit([X_seq, X_ticker], y, batch_size=32, epochs=10)
model.save("multi_stock_model_embedded.h5")
print("Model saved as multi_stock_model_embedded.h5")