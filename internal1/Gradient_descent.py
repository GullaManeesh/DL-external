# Comparison of Different Gradient Descent Techniques on MNIST

import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255
x_test = x_test / 255


# Model creation
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),

        Dense(
            128,
            activation='relu',
            kernel_initializer='glorot_uniform'
        ),

        Dense(
            10,
            activation='softmax',
            kernel_initializer='glorot_uniform'
        )
    ])

    return model


# ============================================================
# 1. Batch Gradient Descent (BGD)
# ============================================================

print("\n========== Batch Gradient Descent ==========")

model = create_model()

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=len(x_train),
    verbose=1
)

end = time.time()

print("\nTime taken by BGD:", end - start)
print("Train Accuracy of BGD:", history.history['accuracy'][-1] * 100)
print("Train Loss of BGD:", history.history['loss'][-1])

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('BGD Training Loss')
plt.legend()
plt.show()


# ============================================================
# 2. Mini Batch Gradient Descent (MBGD - batch_size=500)
# ============================================================

print("\n========== Mini Batch Gradient Descent (500) ==========")

model = create_model()

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=500,
    verbose=1
)

end = time.time()

print("\nTime taken by MBGD:", end - start)
print("Train Accuracy of MBGD:", history.history['accuracy'][-1] * 100)
print("Train Loss of MBGD:", history.history['loss'][-1])

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MBGD (500) Training Loss')
plt.legend()
plt.show()


# ============================================================
# 3. Mini Batch Gradient Descent (MBGD - batch_size=50)
# ============================================================

print("\n========== Mini Batch Gradient Descent (50) ==========")

model = create_model()

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=50,
    verbose=1
)

end = time.time()

print("\nTime taken by MBGD:", end - start)
print("Train Accuracy of MBGD:", history.history['accuracy'][-1] * 100)
print("Train Loss of MBGD:", history.history['loss'][-1])

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MBGD (50) Training Loss')
plt.legend()
plt.show()


# ============================================================
# 4. Momentum Gradient Descent
# ============================================================

print("\n========== Momentum Gradient Descent ==========")

model = create_model()

model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=50,
    verbose=1
)

end = time.time()

print("\nTime taken by Momentum GD:", end - start)
print("Train Accuracy of Momentum GD:", history.history['accuracy'][-1] * 100)
print("Train Loss of Momentum GD:", history.history['loss'][-1])

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Momentum GD Training Loss')
plt.legend()
plt.show()


# ============================================================
# 5. Adam Optimizer
# ============================================================

print("\n========== Adam Optimizer ==========")

model = create_model()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=500,
    verbose=1
)

end = time.time()

print("\nTime taken by Adam:", end - start)
print("Train Accuracy of Adam:", history.history['accuracy'][-1] * 100)
print("Train Loss of Adam:", history.history['loss'][-1])

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Adam Training Loss')
plt.legend()
plt.show()


# ============================================================
# 6. Stochastic Gradient Descent (SGD)
# ============================================================

print("\n========== Stochastic Gradient Descent ==========")

model = create_model()

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=1,
    verbose=1
)

end = time.time()

print("\nTime taken by SGD:", end - start)
print("Train Accuracy of SGD:", history.history['accuracy'][-1] * 100)
print("Train Loss of SGD:", history.history['loss'][-1])

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('SGD Training Loss')
plt.legend()
plt.show()
