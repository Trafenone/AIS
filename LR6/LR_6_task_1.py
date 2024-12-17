import pandas as pd

data = pd.DataFrame({
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
})

frequency_tables = {}

for column in ["Outlook", "Humidity", "Wind"]:
    frequency_tables[column] = pd.crosstab(data[column], data["Play"])

likelihood_tables = {}

for column in ["Outlook", "Humidity", "Wind"]:
    likelihood_tables[column] = {
        "Yes": frequency_tables[column]["Yes"] / frequency_tables[column]["Yes"].sum(),  # P(Attribute|Yes)
        "No": frequency_tables[column]["No"] / frequency_tables[column]["No"].sum()       # P(Attribute|No)
    }

for column in ["Outlook", "Humidity", "Wind"]:
    print(f"Частотна таблиця для {column}:")
    print(frequency_tables[column], "\n")
    print(f"Таблиця правдоподібності для {column}:")
    print("P(Yes|{0})".format(column))
    print(likelihood_tables[column]["Yes"])
    print("P(No|{0})".format(column))
    print(likelihood_tables[column]["No"], "\n")
