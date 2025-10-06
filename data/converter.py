import pandas as pd
import os

# Load the CSV
df = pd.read_csv(os.path.join("data","stock_data.csv"))

# Keep only static columns
static_columns = ["Symbol", "Name", "Sector", "Industry", "Country", "IPO Year"]
df_static = df[static_columns]

# Save to JSON
df_static.to_json("stocks.json", orient="records", indent=2)

print("stocks.json created!")
