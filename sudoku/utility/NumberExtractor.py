import numpy as np
import cv2

# import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json, load_model
from .CommonFunctions import show_image

# Load the saved model
json_file = open("model1.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded saved model from disk.")

# loaded_model = load_model("model1.h5")
# print("Loaded saved model from disk.============>")


# evaluate loaded model on test data
def identify_number(image):
    image_resize = cv2.resize(image, (28, 28))  # For plt.imshow
    image_resize_2 = image_resize.reshape(1, 28, 28, 1)
    print("shape===========>".format(image_resize_2.shape))
    # For input to model.predict_classes
    #    cv2.imshow('number', image_test_1)
    predictions = loaded_model.predict(image_resize_2)
    predicted_classes = np.argmax(predictions, axis=1)
    # loaded_model_pred = loaded_model.predict_classes(image_resize_2, verbose=0)
    return predicted_classes[0]


def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450, 450))
    #    cv2.imshow('sudoku', sudoku)

    # split sudoku
    grid = np.zeros([9, 9])
    for i in range(9):
        for j in range(9):
            #            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
            image = sudoku[i * 50 : (i + 1) * 50, j * 50 : (j + 1) * 50]
            #            filename = "images/sudoku/file_%d_%d.jpg"%(i, j)
            #            cv2.imwrite(filename, image)
            if image.sum() > 100000:
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0

            print("[extract_number]:sum[{}][{}]======+> {}".format(i, j, image.sum()))
            print("[extract_number]:grid [{}][{}]: {}".format(i, j, grid[i][j]))
            if i == 0 and grid[i][j] in (1, 8, 7, 9):
                show_image(image, "number")
            if (i, j) in [(5, 7)]:
                show_image(image, "matrix[6][8]")

    return grid.astype(int)
