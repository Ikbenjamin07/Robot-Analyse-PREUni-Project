import cv2
import numpy as np


def main():
    image = load_image()
    contours_raw = get_contours(image)
    contours_filterd = filter_contours(contours_raw)
    angle_contours = get_angle_contours(contours_raw)
    circulaire_contours = get_circulaire_contour(contours_raw)

    # Maak een output afbeelding
    output_image = image.copy()

    or_image = cv2.drawContours(image.copy(), contours_raw, -1, (0, 255, 0), 2)
    fil_image = cv2.drawContours(image.copy(), contours_filterd, -1, (0, 255, 0), 2)
    contoured_image = cv2.drawContours(output_image, circulaire_contours, -1, (0, 255, 0), 2)

    # Toon de originele afbeelding en de afbeelding met gedetecteerde contouren
    cv2.imshow('Original Image', or_image)
    cv2.imshow('image', image)
    cv2.imshow('Contours', contoured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_contours(image):
    # Converteer de afbeelding naar grijswaarden
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gebruik een Gaussiaanse vervaging om ruis te verminderen
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecteer randen met de Canny-randdetectiemethode
    edges = cv2.Canny(blurred, 50, 150)

    # Vind de contouren in de afbeelding
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def filter_contours(contours):
    filtered_contours = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= 30 and height >= 30:
            filtered_contours.append(contour)
    return filtered_contours

def get_angle_contours(contours, point_skip=5):
    angle_contours = []
    for contour in contours:
        list_angles = []
        for i in range(len(contour)):
            # get points from contour to get an angle
            p1 = contour[i][0]
            p2 = contour[(i + point_skip) % len(contour)][0]
            p3 = contour[(i - point_skip) % len(contour)][0]


            # create vectors between the points (p1, p2) (p3, p2)
            v1 = get_vector(p1, p2)
            v2 = get_vector(p1, p3)

            # get angle between v1 and v2
            angle = get_angle(v1, v2)


            # check if angle is in boundaries
            if angle <= 30 and angle != 0:
                if angle_long_enough(contour, p1, i, angle, v1, v2, point_skip) == True:
                    list_angles.append(angle)

        if len(list_angles) > 0:
            angle_contours.append(contour)

    return angle_contours

def angle_long_enough(contour, angle_point, i, original_angle, v1, v2, point_skip, angle_devaition=5, minimal_length=20):
    angle_length = point_skip
    for k in range(point_skip + 1, len(contour)):
        p1 = angle_point
        p2 = contour[(i + k) % len(contour)][0]
        p3 = contour[(i - k) % len(contour)][0]

        v1new = get_vector(p1, p2)
        v2new = get_vector(p1, p3)

        angle_p2 = get_angle(v1, v1new)
        angle_p3 = get_angle(v2, v2new)
        angle = get_angle(v1new, v2new)
        
        if original_angle - angle_devaition < angle_p2 < original_angle + angle_devaition and original_angle - angle_devaition < angle_p3 < original_angle + angle_devaition and original_angle - angle_devaition < angle < original_angle + angle_devaition:
            angle_length += 1
        else: 
            if angle_length >= minimal_length:
                angle_surface = calculate_surface_angle(p1, p2, p3)
                #print(angle_surface)
            break

    if angle_length >= minimal_length:
        return True
    else:
        return False

def get_vector(p1, p2):
    v = np.array(p1) - np.array(p2)
    return v

def get_angle(v1, v2):
    angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))

    angle = (angle ** 2) ** 0.5

    if angle > 180:
        angle = 360 - angle 

    return angle

def get_circulaire_contour(contours, point_skip=3):
    circulaire_contours = []
    for contour in contours:
        list_curves = []
        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i + point_skip) % len(contour)][0] 
            p3 = contour[(i + (point_skip * 2)) % len(contour)][0]

            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p1)

            a1 = np.degrees(np.arctan2(v1[1], v1[0]))
            a2 = np.degrees(np.arctan2(v2[1], v2[0]))

            if 90 <= a1 < 360:
                if a2 > a1:
                    a2 = a1 - 90 -(a2 - a1)
            if 0 <= a1 < 90:
                if a2 > a1:
                    a2 = a1 - 90 -(a2 - a1)*-1
            if 270 < a1 < 360:
                if a2 < (a1 - 270):
                    a2 = (a1 - 90)+(a1 - 360) - a2
            
            if a2 <= a1 and a2 >= a1 - 180 and abs(a1 - a2) != 0:
                circle_length = circle_long_enough(contour, i, point_skip, p1, v1, a1)
                print(circle_length)
                if circle_length >= 10:
                    list_curves.append(i)
        #print(list_curves)
    

def circle_long_enough(contour, i, point_skip, p1, v1, a1):
    circle_lenght = point_skip
    for k in range(point_skip + 1, len(contour)): 
        p2 = contour[(i + k) % len(contour)][0]
        v2 = get_vector(p1, p2)
        a2 = get_angle(v1, v2)

        if 90 <= a1 < 360:
            if a2 > a1:
                a2 = a1 - 90 -(a2 - a1)
        if 0 <= a1 < 90:
            if a2 > a1:
                a2 = a1 - 90 -(a2 - a1)*-1
        if 270 < a1 < 360:
            if a2 < (a1 - 270):
                a2 = (a1 - 90)+(a1 - 360) - a2
        
        if a2 <= a1 and a2 >= a1 - 180 and abs(a1 - a2) != 0:
            circle_lenght += 1
        else:
            print(circle_lenght)
    return circle_lenght


def load_image():
    # Laad de afbeelding
    image = cv2.imread('img/Da+Vinci.jpg')
    if image is None:
        print("Afbeelding niet gevonden!")
        exit()
    return image 

def calculate_surface_angle(p1, p_end_1, p_end_2):
    surface1 = (((p1[0] - p_end_1[0]) ** 2) ** 0.5) * (((p1[1] - p_end_1[1]) ** 2) ** 0.5) * 0.5
    surface2 = (((p1[0] - p_end_2[0]) ** 2) ** 0.5) * (((p1[1] - p_end_2[1]) ** 2) ** 0.5) * 0.5
    surface_total = surface1 + surface2
    return surface_total










if __name__ == "__main__":
    main()