import cv2
import sys


def output(a):
    sys.stdout.write(str(a))


def display_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                output(".")
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(" |")

            if j != 8:
                output("  ")
        output("\n")
        if (i + 1) % 3 == 0 and i < 8:
            output("--------+----------+---------\n")


def show_image(img, name="img"):
    """Shows an image until any key is pressed"""
    print(type(img))
    print(img.shape)
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    processed_img = cv2.resize(img, (500, 500))  # Resize image
    cv2.imshow(name, img)
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows
    return img
