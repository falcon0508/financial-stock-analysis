import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

function StockChart({ ticker }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    if (!ticker) return;

    const fetchHistory = async () => {
      try {
        const res = await axios.get(`https://financial-stock-analysis-backend.onrender.com/historical?ticker=${ticker}`);
        console.log("Fetched historical data:", res.data);

        // Map to numbers and ensure proper keys
        const formatted = res.data.map((d) => ({
          Date: d.Date,
          Close: Number(d.Close),
        }));

        setData(formatted);
      } catch (error) {
        console.error("Error fetching historical data:", error);
        setData([]);
      }
    };

    fetchHistory();
  }, [ticker]);

  if (!data.length) {
    return <p className="mt-4 text-gray-500">Loading historical data...</p>;
  }

  return (
    <div className="w-full flex justify-center mt-6" style={{
        width: "200%",          // 2x wider than container
        height: 400,
        position: "relative",
        left: "50%",            // move left edge to the middle
        transform: "translateX(-50%)", // shift left by half of its width
      }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="Date" tick={{ fontSize: 12 }} interval={Math.floor(data.length / 10)} />
          <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
          <Tooltip />
          <Line type="monotone" dataKey="Close" stroke="#8884d8" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default StockChart;
