import cv2
import face_recognition
import numpy as np
from PIL import Image
import os
import random

def main():
    file_path = 'img/aibo.jpg'
    image = load_image(file_path)
    contours = get_contours(image)
    filtered_contours = remove_double(contours)
    
    output_image = image.copy()

    areas = []
    for contour in filtered_contours:
            areas.append(cv2.contourArea(contour))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.polylines(output_image, [contour], isClosed=False, color=color, thickness=2)
    print(areas)
    cv2.imshow(f'Circle Image: {file_path}', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows


def remove_double(contours):
    new_contours = []
    for contour in contours: 
        if cv2.contourArea(contour) < 5:
            new_contour = contour[:round(len(contour)/2)]
            new_contours.append(new_contour)
            print(new_contour)
            return
        else:
            new_contours.append(contour)
    return new_contours

def load_image(img_path):
    # Laad de afbeelding
    image = cv2.imread(img_path)
    if image is None:
        print("Afbeelding niet gevonden!")
        exit()
    return image

def get_contours(image):
    # Converteer de afbeelding naar grijswaarden
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gebruik een Gaussiaanse vervaging om ruis te verminderen
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detecteer randen met de Canny-randdetectiemethode
    edges = cv2.Canny(blurred, 50, 150)

    # Vind de contouren in de afbeelding
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

if __name__ == "__main__":
    main()