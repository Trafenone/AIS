import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Завантаження та підготовка даних з текстового файлу
input_file = 'income_data.txt'
X = []  # Вхідні дані (ознаки)
y = []  # Цільові значення (мітки класів)
count_class1 = 0  # Лічильник для класу 0
count_class2 = 0  # Лічильник для класу 1
max_datapoints = 2500  # Максимальна кількість зразків для кожного класу

# Читання та обробка даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:  # Пропускаємо рядки з пропущеними значеннями
            continue
        # Розділяємо рядок на компоненти та визначаємо клас
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)  # Клас 0 для '<=50K'
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)  # Клас 1 для '>50K'
            count_class2 += 1

# Перетворення списків на масиви numpy
X = np.array(X)
y = np.array(y)

# Кодування категоріальних змінних
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():  # Якщо значення числове, не кодуємо
        X_encoded[:, i] = X[:, i].astype(float)
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])  # Кодуємо категоріальні змінні
        label_encoder.append(encoder)

# Перетворюємо X на масив чисел
X = X_encoded.astype(float)

# Розподіл даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Список алгоритмів для класифікації
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Оцінка моделей за допомогою крос-валідації
results = []
names = []
for name, model in models:
    # Крос-валідація з 10 згинами
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

# Візуалізація результатів за допомогою діаграми розмаху
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів')
plt.show()

# Навчання та оцінка кожної моделі на тестових даних
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'{name} - Точність: {accuracy:.2f}, F1-Оцінка: {f1:.2f}, Точність: {precision:.2f}, Повторюваність: {recall:.2f}')
