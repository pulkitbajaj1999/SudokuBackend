from django.http import HttpResponse, JsonResponse
import numpy as np


from .utility.SudokuExtractor import extract_sudoku

from .utility.NumberExtractor import extract_number

# from .utility.SolveSudoku import sudoku_solver
from .utility.CommonFunctions import display_sudoku


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def process(request):
    print("processing-start==========>")
    sudoku2Path = "images/sudoku3.png"
    image = extract_sudoku(sudoku2Path)
    grid = extract_number(image)
    print("Sudoku:")
    display_sudoku(grid.tolist())
    # solution = sudoku_solver(grid)
    # print("Solution:")
    # print(solution)
    # display_sudoku(solution.tolist())
    return JsonResponse({"success": True})
