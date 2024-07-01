from django.http import HttpResponse, JsonResponse
import numpy as np
import cv2
import base64


from .utility.SudokuExtractor import extract_sudoku

from .utility.NumberExtractor import extract_number

# from .utility.SolveSudoku import sudoku_solver
from .utility.CommonFunctions import display_sudoku


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def process(request):
    if request.method == "POST":
        uploaded_image = request.FILES["file"]
        image_array = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    elif request.method == "GET":
        print("processing default: sudoku4.jpg==============>")
        image = cv2.imread("images/sudoku4.jpg", cv2.IMREAD_GRAYSCALE)

    print("extracting-sudoku================>")
    processed_image = extract_sudoku(image, debug=False)
    ret, jpeg = cv2.imencode(".jpg", processed_image)
    base64_string = base64.b64encode(jpeg).decode("utf-8")
    # response_bytes = jpeg.tobytes()

    print("extracting-number==================>")
    grid = extract_number(processed_image)
    print("display-Sudoku:============>")
    display_sudoku(grid.tolist())
    return JsonResponse(
        {
            "success": True,
            "data": {"processedImage": base64_string, "matrix": grid.tolist()},
        }
    )
    return HttpResponse(response_bytes, content_type="image/jpeg")

    # solution = sudoku_solver(grid)
    # print("Solution:")
    # print(solution)
    # display_sudoku(solution.tolist())
