import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

plt.imshow(X_train[0])
X_train = X_train/255
X_test = X_test/255

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Valid Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()


plt.imshow(X_test[10])
print(y_pred[10])



############
# 1. load image
img = cv2.imread("/content/digit_9.png")

# 2. grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. invert (MNIST style)
img = cv2.bitwise_not(img)

# 4. resize to 28x28
img = cv2.resize(img, (28, 28))

# 5. normalize
img = img / 255.0

# 6. reshape for model
img = img.reshape(1, 28, 28)

plt.imshow(img[0])
y_prob = model.predict(img)
print(y_prob)
argmax = y_prob.argmax()
print(argmax)
