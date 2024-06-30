# chatgpt model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape data to fit the model (add a single channel for grayscale)
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))


# Initialize a Sequential model
model = Sequential()

# Add a 2D convolutional layer with 32 filters, each of size 3x3, relu activation
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))

# Add a max pooling layer with pool size 2x2
model.add(MaxPooling2D((2, 2)))

# Add dropout layer with dropout rate of 0.25
model.add(Dropout(0.25))

# Flatten the 2D arrays to a 1D array for fully connected layers
model.add(Flatten())

# Add fully connected layer with 128 nodes and relu activation
model.add(Dense(128, activation="relu"))

# Add dropout layer with dropout rate of 0.5
model.add(Dropout(0.5))

# Add output layer with 10 nodes (one for each class) and softmax activation
model.add(Dense(10, activation="softmax"))


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# Save the model to disk
model.save("model3.h5")

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]*100:.2f}%")
