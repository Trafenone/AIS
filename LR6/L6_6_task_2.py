import pandas as pd

data = [
    ["Sunny", "High", "Weak", "No"],
    ["Sunny", "High", "Strong", "No"],
    ["Overcast", "High", "Weak", "Yes"],
    ["Rain", "High", "Weak", "Yes"],
    ["Rain", "Normal", "Weak", "Yes"],
    ["Rain", "Normal", "Strong", "No"],
    ["Overcast", "Normal", "Strong", "Yes"],
    ["Sunny", "High", "Weak", "No"],
    ["Sunny", "Normal", "Weak", "Yes"],
    ["Rain", "High", "Weak", "Yes"],
    ["Sunny", "Normal", "Strong", "Yes"],
    ["Overcast", "High", "Strong", "Yes"],
    ["Overcast", "Normal", "Weak", "Yes"],
    ["Rain", "High", "Strong", "No"]
]

columns = ["Outlook", "Humidity", "Wind", "Play"]
df = pd.DataFrame(data, columns=columns)

frequency_table = df.groupby(["Outlook", "Play"]).size().unstack()
print("Частотна таблиця:")
print(frequency_table)

total_yes = len(df[df["Play"] == "Yes"])
total_no = len(df[df["Play"] == "No"])
total = len(df)

prob_yes = total_yes / total
prob_no = total_no / total

print(f"\nЙмовірність 'Yes': {prob_yes}")
print(f"Ймовірність 'No': {prob_no}")

prob_outlook_rain_yes = len(df[(df["Outlook"] == "Rain") & (df["Play"] == "Yes")]) / total_yes
prob_humidity_high_yes = len(df[(df["Humidity"] == "High") & (df["Play"] == "Yes")]) / total_yes
prob_wind_strong_yes = len(df[(df["Wind"] == "Strong") & (df["Play"] == "Yes")]) / total_yes

prob_outlook_rain_no = len(df[(df["Outlook"] == "Rain") & (df["Play"] == "No")]) / total_no
prob_humidity_high_no = len(df[(df["Humidity"] == "Normal") & (df["Play"] == "No")]) / total_no
prob_wind_strong_no = len(df[(df["Wind"] == "Strong") & (df["Play"] == "No")]) / total_no

prob_yes_given_conditions = prob_yes * prob_outlook_rain_yes * prob_humidity_high_yes * prob_wind_strong_yes
prob_no_given_conditions = prob_no * prob_outlook_rain_no * prob_humidity_high_no * prob_wind_strong_no

print(f"\nЙмовірність 'Yes': {prob_yes_given_conditions}")
print(f"Ймовірність 'No': {prob_no_given_conditions}")

if prob_yes_given_conditions > prob_no_given_conditions:
    print("\nМатч відбудеться!")
else:
    print("\nМатч не відбудеться!")
