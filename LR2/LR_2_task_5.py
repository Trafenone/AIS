import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO

# Завантажуємо набір даних Iris
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Розподіляємо дані на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ініціалізуємо та тренуємо класифікатор Ridge
ridge_classifier = RidgeClassifier(tol=1e-2, solver="sag")
ridge_classifier.fit(X_train, y_train)

# Виконання прогнозу для тестових даних
y_pred = ridge_classifier.predict(X_test)

# Виведення основних метрик оцінки
print('Точність:', np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Точність за класами:', np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
print('Відновлення:', np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))
print('F1-Оцінка:', np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))
print('Cohen Kappa:', np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Кореляція Меттіюса:', np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))

# Детальний класифікаційний звіт
print('\n\t\tКласифікаційний звіт:\n', metrics.classification_report(y_test, y_pred))

# Матриця плутанини
conf_matrix = confusion_matrix(y_test, y_pred)

# Налаштування стилю для кращої візуалізації
sns.set()

# Відображення матриці плутанини у вигляді теплової карти
sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Правильні мітки')
plt.ylabel('Прогнозовані мітки')

# Збереження теплової карти як зображення
plt.savefig("Confusion_Matrix.jpg")

# Збереження графіка у форматі SVG
svg_buffer = BytesIO()
plt.savefig(svg_buffer, format="svg")
