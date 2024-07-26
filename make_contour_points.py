import cv2
import numpy as np
from PIL import Image

def detect_non_transparent_contours(image_path):
    # Step 1: Load the image with transparency
    image = Image.open(image_path)
    image = image.convert("RGBA")  # Ensure image is in RGBA format to get the alpha channel
    cv_image = np.array(image)  # Convert PIL image to an OpenCV image
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)

    # Step 2: Extract the alpha channel
    alpha_channel = cv_image[:, :, 3]

    # Step 3: Apply a threshold to identify non-transparent areas
    _, thresholded = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours in the thresholded alpha channel
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Optional: Draw contours on a blank canvas
    # blank_canvas = np.zeros(cv_image.shape, dtype=np.uint8)
    # cv2.drawContours(blank_canvas, contours, -1, (0, 255, 0, 255), 2)

    # Step 5: Draw contours on the original image
    cv2.drawContours(cv_image, contours, -1, (0, 255, 0, 255), 2)

    # Convert back to PIL image to display/save
    final_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))

    # Display or save the final image
    final_image.show()
    # final_image.save('output_path.png')

# Replace 'path_to_your_image.png' with the path to your PNG image with transparency
detect_non_transparent_contours('test.png')