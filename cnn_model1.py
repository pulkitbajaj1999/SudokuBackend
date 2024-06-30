# https://github.com/aakashjhawar/SolveSudoku/blob/master/cnn.py
# Larger CNN for the MNIST Dataset
import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K

# import matplotlib.pyplot as plt

K.set_image_data_format("channels_last")

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Total no of Images: ", X_train.shape[0])
print("Size of Image:", X_train.shape[1:])
print("Total no of labels:", y_train.shape)
print("y_test[0:10]=====>{}", y_test[0:10])

# Reshape to be samples*pixels*width*height
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One Hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))  # num_classes = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))


# - - - - - - TESTING multiple image - - - - - - - - - -
test_images = X_test[0:10]
print("Test images shape: {}".format(test_images.shape))

for i, test_image in enumerate(test_images):
    org_image = test_image
    test_image = test_image.reshape(1, 28, 28, 1)
    predictions = model.predict(test_image)
    predicted_classes = np.argmax(predictions, axis=-1)
    print("predicted_classes for {}: {}".format(i, predicted_classes[0]))
    # plt.subplot(220 + i)
    # plt.axis("off")
    # plt.title("Predicted digit: {}".format(prediction[0]))
    # plt.imshow(org_image, cmap=plt.get_cmap("gray"))

# plt.show()


# - - - - - - - SAVE THE MODEL - - - - - - - -

# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")
