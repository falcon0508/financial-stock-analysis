from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
from keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://financial-stock-analysis.onrender.com"}})

# Load multi-stock model
model = load_model("multi_stock_model_gpu.h5")

# Load ticker encoder
ticker_encoder = joblib.load("ticker_encoder.pkl")

# Folder where scalers are saved
SCALER_FOLDER = "scalers"

SEQUENCE_LENGTH = 60

@app.route("/predict")
def predict():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400
    
    if ticker not in ticker_encoder.classes_:
        return jsonify({"error": f"Ticker {ticker} not supported"}), 404

    try:
        data = yf.download(ticker, period="90d", auto_adjust=True)
        close_prices = data["Close"].values[-SEQUENCE_LENGTH:].reshape(-1,1)
        if len(close_prices) < SEQUENCE_LENGTH:
            return jsonify({"error": "Not enough data for this ticker"}), 400

        # Load scaler
        scaler_path = os.path.join(SCALER_FOLDER, f"scaler_{ticker}.pkl")
        if not os.path.exists(scaler_path):
            return jsonify({"error": f"No scaler found for {ticker}"}), 404
        scaler = joblib.load(scaler_path)

        scaled_data = scaler.transform(close_prices)
        X_seq = np.reshape(scaled_data, (1, SEQUENCE_LENGTH, 1))
        X_ticker = np.array([[ticker_encoder.transform([ticker])[0]]])

        pred_scaled = model.predict([X_seq, X_ticker])[0,0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0,0]

        return jsonify({"prediction": float(pred_price)})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/historical")
def historical():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        data = yf.download(ticker, period="180d", auto_adjust=True)
        if data.empty:
            return jsonify({"error": f"No historical data found for {ticker}"}), 404

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["_".join(col).strip() for col in data.columns.values]

        # Find the Close column dynamically
        close_cols = [col for col in data.columns if "Close" in col]
        if not close_cols:
            return jsonify({"error": f"No Close column found for {ticker}"}), 500

        close_col = close_cols[0]  # pick the first matching Close column
        close_series = data[[close_col]].dropna().reset_index()
        close_series.rename(columns={close_col: "Close"}, inplace=True)

        # Convert Date to string, Close to float
        history = [
            {"Date": d.strftime("%Y-%m-%d"), "Close": float(c)}
            for d, c in zip(close_series["Date"], close_series["Close"])
        ]

        return jsonify(history)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# Backend debugging

import traceback

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()  # this prints full traceback in Render logs
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
