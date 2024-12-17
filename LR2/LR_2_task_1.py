import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# Вхідний файл який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        # Розбиття рядка та відокремлення мітки класу
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)  # Клас 0 для '<=50K'
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)  # Клас 1 для '>50K'
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)І
y = np.array(y)

# Кодування рядкових даних у числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i].astype(float)
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)
# Відокремлення ознак і міток
X = X_encoded.astype(float)
# Створення SVM-класифікатора з лінійним ядром
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
# Розбиття даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Навчання класифікатора
classifier.fit(X_train, y_train)
# Прогнозування результатів для тестового набору
y_test_pred = classifier.predict(X_test)
# Обчислення F1-міри та інших показників якості
f1 = f1_score(y_test, y_test_pred, average='weighted')
accuracy = accuracy_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred, average='weighted')
precision = precision_score(y_test, y_test_pred, average='weighted')
print("F1 score: {:.2f}%".format(f1 * 100))
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("Precision: {:.2f}%".format(precision * 100))
# Передбачення для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']
# Кодування тестової точки даних
input_data_encoded = np.empty(len(input_data))
count = 0  # Лічильник для label_encoder
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = label_encoder[count].transform([item])[0]
        count += 1  # Збільшити лічильник тільки для текстових даних
# Перетворення у numpy масив
input_data_encoded = np.array(input_data_encoded)
# Використання класифікатора для передбачення результату
predicted_class = classifier.predict([input_data_encoded])
predicted_label = label_encoder[-1].inverse_transform(predicted_class)
print("Predicted class for the input data:", predicted_label[0])
