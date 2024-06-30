# https://www.kaggle.com/code/manthansolanki/image-classification-with-mnist-dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

np.random.seed(42)  # This allows us to reproduce the results from our script
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical  # help us to transform our data later


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


print("Total no of Images: ", X_train.shape[0])
print("Size of Image:", X_train.shape[1:])
print("Total no of labels:", y_train.shape)


# look input data

plt.imshow(X_train[0], cmap=plt.get_cmap("gray"))  # cmap - convert image into grascale
print("Label:", y_train[0])


X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print(X_train.shape, X_test.shape)


X_train = X_train / 255
X_test = X_test / 255

# print(X_train[0])
X_train.shape


# One-hot encoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)


num_classes = y_test.shape[1]
num_pixels = 784


# define baseline model


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=num_pixels, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model


# build the model
model = baseline_model()
model.summary()


opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)


# Save the model to disk
model.save("model4.h5")

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]*100:.2f}%")


# predict
# img_width, img_height = 28, 28

# ii = cv2.imread("../input/mnistpredict/3.png")
# gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
# # print(gray_image)
# plt.imshow(gray_image,cmap='Greys')
# plt.show()
# # gray_image.shape
# x = np.expand_dims(gray_image, axis=0)
# x = x.reshape((1, -1))

# preds = model.predict_classes(x)
# prob = model.predict_proba(x)


# print('Predicted value is ',preds[0])
# print('Probability across all numbers :', prob)
