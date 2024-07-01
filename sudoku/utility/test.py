import cv2
import numpy as np


def find_largest_feature_org(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if (
                img.item(y, x) == 255 and x < width and y < height
            ):  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if (
                    area[0] > max_area
                ):  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros(
        (height + 2, width + 2), np.uint8
    )  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype="float32"), seed_point


def find_largest_feature_1(image, scan_tl=None, scan_br=None):
    # Convert the image to grayscale if it's not already in
    gray = image.copy()

    # Apply binary thresholding (adjust threshold value as needed)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    max_area = -1
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    bbox = None
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = [[x, y], [x + w, y + h]]
    else:
        bbox = [[0, 0], [0, 0]]

    # Create a mask for the largest contour
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)  # Draw filled contour on mask

    # Extract the largest feature using the mask
    largest_feature = cv2.bitwise_and(image, image, mask=mask)
    return largest_feature, np.array(bbox, dtype="float32"), None


def findLargestFeature_2(image, scan_tl=None, scan_br=None):
    # Convert the image to grayscale if it's not already in grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply binary thresholding to create a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours to get the largest connected component
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the largest feature
    largest_feature = None
    largest_area = -1

    # Iterate through contours to find the largest one
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Extract the largest feature using the bounding box
            largest_feature = image[y : y + h, x : x + w]

    return largest_feature, np.array([[x, y], [x + w, y + h]], dtype="float32"), None
