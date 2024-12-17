import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 2500

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

X = np.array(X)
y = np.array(y)
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i].astype(float)
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)
X = X_encoded.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Функція для оцінки роботи SVM з різними ядрами
def assess_svm_performance(kernel_type):
    # Ініціалізація класифікатора
    svm_model = SVC(kernel=kernel_type, random_state=0)
    svm_model.fit(X_train, y_train)  # Навчання моделі
    predictions = svm_model.predict(X_test)  # Прогнозування на тестовому наборі

    # Розрахунок метрик
    f1_metric = f1_score(y_test, predictions, average='weighted')
    accuracy_metric = accuracy_score(y_test, predictions)
    recall_metric = recall_score(y_test, predictions, average='weighted')
    precision_metric = precision_score(y_test, predictions, average='weighted')

    # Виведення результатів
    print(f"Kernel: {kernel_type}")
    print("F1 Score: {:.2f}%".format(f1_metric * 100))
    print("Accuracy: {:.2f}%".format(accuracy_metric * 100))
    print("Recall: {:.2f}%".format(recall_metric * 100))
    print("Precision: {:.2f}%".format(precision_metric * 100))
    print()

# Виконання оцінки для трьох типів ядер
assess_svm_performance('poly')
assess_svm_performance('rbf')
assess_svm_performance('sigmoid')
