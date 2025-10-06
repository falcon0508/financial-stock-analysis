from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ===============================
# Load TFLite model
# ===============================
TFLITE_MODEL_PATH = "multi_stock_model_dense.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load ticker encoder
ticker_encoder = joblib.load("ticker_encoder.pkl")

# Folder where scalers are saved
SCALER_FOLDER = "scalers"
SEQUENCE_LENGTH = 30

# ===============================
# Routes
# ===============================
@app.route("/", strict_slashes=False)
def index():
    return jsonify({
        "status": "ok",
        "message": "Financial Stock Analysis Backend is running."
    })

@app.route("/predict")
def predict():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    if ticker not in ticker_encoder.classes_:
        return jsonify({"error": f"Ticker {ticker} not supported"}), 404

    try:
        # Download recent data
        data = yf.download(ticker, period=f"{SEQUENCE_LENGTH}d", auto_adjust=True)
        close_prices = data["Close"].values.reshape(-1,1)
        if len(close_prices) < SEQUENCE_LENGTH:
            return jsonify({"error": f"Not enough data for {ticker} (need {SEQUENCE_LENGTH} days)"}), 400

        # Load stock scaler
        scaler_path = os.path.join(SCALER_FOLDER, f"scaler_{ticker}.pkl")
        if not os.path.exists(scaler_path):
            return jsonify({"error": f"No scaler found for {ticker}"}), 404
        scaler = joblib.load(scaler_path)

        # Scale input sequence
        scaled_data = scaler.transform(close_prices)
        X_seq = np.reshape(scaled_data, (1, SEQUENCE_LENGTH, 1)).astype(np.float32)

        # Prepare ticker input: repeat ticker index for each sequence step
        ticker_index = ticker_encoder.transform([ticker])[0]
        X_ticker = np.full((1, SEQUENCE_LENGTH), ticker_index, dtype=np.float32)

        # Set tensors for TFLite
        interpreter.set_tensor(input_details[0]['index'], X_seq)
        interpreter.set_tensor(input_details[1]['index'], X_ticker)

        # Run inference
        interpreter.invoke()
        pred_scaled = interpreter.get_tensor(output_details[0]['index'])[0,0]
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

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["_".join(col).strip() for col in data.columns.values]

        close_cols = [col for col in data.columns if "Close" in col]
        if not close_cols:
            return jsonify({"error": f"No Close column found for {ticker}"}), 500

        close_col = close_cols[0]
        close_series = data[[close_col]].dropna().reset_index()
        close_series.rename(columns={close_col: "Close"}, inplace=True)

        history = [
            {"Date": d.strftime("%Y-%m-%d"), "Close": float(c)}
            for d, c in zip(close_series["Date"], close_series["Close"])
        ]

        return jsonify(history)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# ===============================
# Backend debugging
# ===============================
import traceback
@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
