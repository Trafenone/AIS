import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Завантаження даних з URL
url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

# Видалення рядків з пропущеними значеннями
data = data.dropna()

# Вибір необхідних стовпців для моделювання
features = ["train_type", "train_class", "fare"]

# Підготовка вхідних даних
X = data[features].apply(lambda col: pd.factorize(col)[0])  # Перетворення категоріальних змінних у числові
y = data["price"] > data["price"].median()  # Цільова змінна: чи є ціна вище медіани

# Створення і тренування моделі наївного байєсівського класифікатора
model = GaussianNB()
model.fit(X, y)

# Прогнозування на основі навченого класифікатора
predictions = model.predict(X)
print("\nПерші 10 передбачених значень:")
print(predictions[:10])

# Описова статистика для ціни
print("\nОписова статистика цін:")
print(data["price"].describe())
