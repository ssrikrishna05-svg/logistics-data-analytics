import pandas as pd

# Load logistics dataset
df = pd.read_csv("logistics_data.csv")

# Display first five records
print(df.head())

# Calculate on-time delivery rate
total = len(df)
on_time = (df["Delivery_Status"] == "On Time").sum()

rate = (on_time / total) * 100
print("On-Time Delivery Rate:", rate)

# Calculate average delivery time
avg_time = df["Delivery_Time"].mean()
print("Average Delivery Time:", avg_time)
