import cv2
import matplotlib.pyplot as plt
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


def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
