# https://prasad-jayanti.medium.com/image-classification-with-mnist-data-286003b056cb
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from PIL import Image
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def get_data():
    """
    This is data loader, it reads and normalizes the data and also creates training
    and test data set. It is recommended to write this module separetly so that if the
    data source changes, we can update the program easily.

    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255

    return x_train, y_train, x_test, y_test


class CNN_Clsf:
    """
    It is useful to write a machine learning model in terms of an Object with
    properties and methods. Note that here we have a predict method for inference but it need not be
    a part of the object. We will have a separte prediction function also.

    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(
        self,
    ):
        """
        This is the place wheere can build our model.
        """

        input_data = layers.Input(shape=self.input_shape, name="Input-Layer")
        x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv2D-I")(
            input_data
        )
        x = layers.Conv2D(64, (3, 3), activation="relu", name="Conv2D-II")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool")(x)
        x = layers.Dropout(0.25, name="Dropout-I")(x)
        x = layers.Flatten(name="Flatten")(x)
        x = layers.Dense(128, activation="relu", name="Output-Dense")(x)
        x = layers.Dropout(0.5, name="Dropout-II")(x)
        output_data = layers.Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=input_data, outputs=output_data, name="Conv2D-Model")
        print(self.model.summary())

    def fit_model(self, x_train, y_train, epochs):
        """
        This is the training part and it takes maximum time/computation. Note that here
        we need to specify the number of epochs for training as well as other parameters.
        """

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = keras.optimizers.RMSprop()

        model_checkpoint = ModelCheckpoint(
            "model2.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )

        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[model_checkpoint],
        )
        return history

    def predict(self, x_test):
        """
        We can use inbuilt prediction method but we must have it separately also.
        """

        return self.model.predict(x_test)


def plot_accuracy(history):

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    axs[1].plot(history.history["accuracy"], "-o", label="Training")
    axs[1].plot(history.history["val_accuracy"], "-o", label="Validation")
    axs[0].plot(history.history["loss"], "-o", label="Training")
    axs[0].plot(history.history["val_loss"], "-o", label="Validation")

    axs[0].legend()
    axs[1].legend()
    plt.legend()
    plt.show()


def plot_data(x_train):
    rand_ids = np.random.randint(1, x_train.shape[0], [16])
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axs.flat):
        ax.imshow(x_train[i, :, :])
    plt.show()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()

    plot_data(x_train)

    num_classes = 10
    input_shape = (28, 28, 1)

    M = CNN_Clsf(input_shape, num_classes)
    history = M.fit_model(x_train, y_train, 10)

    plot_accuracy(history)

    test_scores = M.model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    y_predict = M.predict(x_test)
    y_hat = [np.argmax(y_predict[i]) for i in range(0, len(y_predict))]

    cm = confusion_matrix(y_test, y_hat)

    print("confudion matrix\n", cm)

    p = precision_score(y_test, y_hat, average="micro")
    r = recall_score(y_test, y_hat, average="micro")

    print("precision =", p)
    print("recall =", r)
