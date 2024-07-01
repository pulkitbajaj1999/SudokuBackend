
## based on [SolveSudoku](https://github.com/aakashjhawar/SolveSudoku/tree/master)


# SolveSudoku 
SolveSudoku extract and solve sudoku from image. It uses a collection of image processing techniques and Convolution Neural Network for training and recognition of characters.
CNN is trained on MNIST dataset to detect digits.

## Blog
Checkout the **articles on SolveSudoku** on Medium.com 

[Sudoku Solver using OpenCV and DL — Part 1](https://medium.com/@aakashjhawar/sudoku-solver-using-opencv-and-dl-part-1-490f08701179)

[Sudoku Solver using OpenCV and DL — Part 2](https://medium.com/@aakashjhawar/sudoku-solver-using-opencv-and-dl-part-2-bbe0e6ac87c5)

 
## Prerequisites

- Python 3.5 or above
- OpenCV
- tensorflow

## Getting Started
How to use
1. clone the repo
```
git clone https://github.com/aakashjhawar/SolveSudoku.git
```
2. set up python virtual environment in the repo
3. install the requirements2.txt
```
pip install -r requirement2.txt
```
5. (optional) train the model specific to your system
```
python cnn_model1.py
```
4. run djnago server
```
python manage.py runserver
```
5. checkout [frontend repo](https://github.com/pulkitbajaj1999/sudoku-frontend)
```
https://github.com/pulkitbajaj1999/sudoku-frontend.git
```
6. after starting both fronend and backend got to browser and open frontend


## Procedure
 > 1. Image preprocessing (Thresholding).
 > 2. Find the largest contour (sudoku square).
 > 3. Get the cordinates of **largest contour**.
 > 4. Crop the image.
 > 5. Perform **Warp perspective** on image
 > 5. Extract each cells from the image by slicing the sudoku grid.
 > 6. Extract the **largest component** in the sudoku image (number).
 > 7. Remove noise in block.
 > 8. Send the number to pre trained Digit Recogition model.
 > 9. Send the grid to Sudoku Solver to perform the final step.
## Working 

#### Input image of Sudoku-
![Input image of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/sudoku.jpg)

#### Image of Sudoku after thresholding-
![Threshold image of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/threshold.jpg)

#### Contour corners of Sudoku-
![Contour of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/contour.jpg)

#### Warp Image-
![Warp of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/warp.jpg)

#### Final output of ExtractSudoku-
![Final image of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/final.jpg)


#### Extracted grid-
![extracted grid](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/extracted_grid.png)

#### Solved grid-
![Solved grid](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/solved_grid.png)



