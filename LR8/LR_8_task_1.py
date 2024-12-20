import numpy as np
import tensorflow as tf

# Визначення гіперпараметрів
n_samples, batch_size, num_steps = 1000, 100, 20000
# Генерація випадкових вхідних даних X та цільових значень y з шумом
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)  # Дані типу float32
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)  # Додавання шуму

# Ініціалізація параметрів моделі (нахил і зміщення)
k = tf.Variable(tf.random.normal((1, 1), dtype=tf.float32), name='slope')  # Нахил
b = tf.Variable(tf.zeros((1,), dtype=tf.float32), name='bias')  # Зміщення

# Визначення моделі
def model(X):
    # Модель лінійної регресії: y = k * X + b
    return tf.matmul(X, k) + b

# Функція втрат: сума квадратів різниці між реальними і передбаченими значеннями
def loss_fn(X, y):
    y_pred = model(X)
    return tf.reduce_sum((y - y_pred) ** 2)

# Оптимізатор градієнтного спуску з низькою швидкістю навчання
optimizer = tf.optimizers.SGD(learning_rate=0.001)

# Основний цикл навчання моделі
display_step = 100
for i in range(num_steps):
    # Відбираємо випадкову підвибірку даних
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    # Обчислення функції втрат за допомогою GradientTape
    with tf.GradientTape() as tape:
        loss_val = loss_fn(X_batch, y_batch)

    # Перевірка наявності NaN у функції втрат
    if tf.reduce_any(tf.math.is_nan(loss_val)):
        print(f"NaN у функції втрат на ітерації {i+1}")
        break

    # Обчислення градієнтів для параметрів
    gradients = tape.gradient(loss_val, [k, b])

    # Перевірка наявності NaN у градієнтах
    if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients):
        print(f"NaN у градієнтах на ітерації {i+1}")
        break

    # Обрізання градієнтів для запобігання вибуху
    gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

    # Оновлення параметрів моделі
    optimizer.apply_gradients(zip(gradients, [k, b]))

    # Виведення результатів кожні 100 ітерацій
    if (i+1) % display_step == 0:
        print(f"Ітерація {i+1}: Похибка={loss_val.numpy():.8f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")
