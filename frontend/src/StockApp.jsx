import React, { useState, useMemo } from "react";
import axios from "axios";
import stockData from "../../data/stock_data.json";
import StockChart from "./StockChart";

function StockApp() {
  const [inputTicker, setInputTicker] = useState("");
  const [ticker, setTicker] = useState("");
  const [prediction, setPrediction] = useState(null);

  // Sort stocks alphabetically for easier selection
  const stocks = useMemo(() => 
  stockData
    .filter(s => s.Symbol) // remove entries with null/undefined Symbol
    .sort((a, b) => a.Symbol.localeCompare(b.Symbol)), [], []);


  const handlePredict = async () => {
    if (!inputTicker) return;

    const upperTicker = inputTicker.toUpperCase();

    // Optional: check if ticker is supported
    const supported = stocks.find((s) => s.Symbol === upperTicker);
    if (!supported) {
      alert(`Ticker ${upperTicker} is not supported.`);
      return;
    }

    try {
      const res = await axios.get(`https://financial-stock-analysis-backend.onrender.com/predict?ticker=${upperTicker}`);
      setPrediction(res.data.prediction);
      setTicker(upperTicker); // only update ticker when predict is clicked
    } catch (error) {
      console.error("Prediction API error:", error);
      setPrediction(null);
      alert("Error predicting ticker. Make sure it has enough data.");
    }
  };

  return (
    <div className="p-6 max-w-lg mx-auto">
      <h1 className="text-2xl font-bold">📈 Stock Forecaster</h1>

      <input
        list="tickers"
        value={inputTicker}
        onChange={(e) => setInputTicker(e.target.value)}
        placeholder="Enter or select a ticker"
        className="border p-2 rounded mt-4 w-full"
      />

      <datalist id="tickers">
        {stocks.map((stock) => (
          <option key={stock.Symbol} value={stock.Symbol}>
            {stock.Symbol} – {stock.Name}
          </option>
        ))}
      </datalist>

      <button
        onClick={handlePredict}
        className="bg-blue-500 text-white px-4 py-2 mt-2 rounded"
      >
        Predict
      </button>

      {prediction !== null && (
        <div className="mt-4 text-lg">
          Next-day predicted price for <b>{ticker}</b>: ${prediction.toFixed(2)}
        </div>
      )}

      {ticker && <StockChart ticker={ticker} />}
    </div>
  );
}

export default StockApp;
