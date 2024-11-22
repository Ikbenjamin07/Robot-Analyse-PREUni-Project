import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Thresholding to separate the object from the background
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def get_contours(thresh_image):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def symmetry_score(image):
    # Calculate vertical and horizontal symmetry
    flipped_vertically = cv2.flip(image, 0)
    #flipped_horizontally = cv2.flip(image, 1)
    
    vertical_symmetry = np.sum(image == flipped_vertically) / image.size
    #horizontal_symmetry = np.sum(image == flipped_horizontally) / image.size
    
    return (vertical_symmetry) / 2  # Average symmetry score

def aspect_ratio_score(contour):
    # Calculate the aspect ratio of the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Score based on closeness to a human-like aspect ratio (assuming full body ~0.5)
    return 1 - abs(aspect_ratio - 0.5)  # Close to 0.5 means more human-like

def human_likeness_score():
    folder_path = 'img/' 
    scores = []
    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Preprocess image and extract contours
        thresh_image = preprocess_image(file_path)
        contours = get_contours(thresh_image)

        if len(contours) == 0:
            print("No contours found in the image.")
            return 0

        # Calculate symmetry score
        symmetry = symmetry_score(thresh_image)

        # Calculate aspect ratio score using the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        aspect_ratio = aspect_ratio_score(main_contour)

        # Combine scores to get a final human-likeness score
        human_likeness = (symmetry + aspect_ratio) / 2

        scores.append(human_likeness)
    return scores

def main():
    human_likeness_score()


if __name__ == "__main__":
    main()

