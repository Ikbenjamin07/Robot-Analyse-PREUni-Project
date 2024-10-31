import cv2
import numpy as np
from PIL import Image
import os

def main():
    image = load_image()
    inverted_image = cv2.bitwise_not(image)
    contours_raw = get_contours(inverted_image)

    filtered_contours = filter(contours_raw)
    potential_heads = get_circulaire_contour(filtered_contours, 50, 999)
    potential_eyes = get_circulaire_contour(filtered_contours, 5, 30)
    sliced_heads = slice_contours(potential_heads)
    merged_heads = merge_contours_with_tolerance(sliced_heads)
    heads, eyes = check_eyes(merged_heads, potential_eyes)
    face_recognised = facial_recognition(image)
    score = 0
    if face_recognised:
        score += 3
    print(len(merged_heads))
    for head in merged_heads:
        contoured_image = cv2.drawContours(image, head, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', contoured_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue
    
    

def load_image():
    # Laad de afbeelding
    image = cv2.imread('img/nao.jpg' )
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

def filter(contours_raw):
    filtered_contours = []
    x_values = [subarray[0, 0] for array in contours_raw for subarray in array]
    y_values = [subarray[0, 1] for array in contours_raw for subarray in array]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    

    rob_y_mid = min_y + (max_y - min_y) / 3
    rob_width = (max_x - min_x)

    left_middle = min_x + rob_width / 2 - 0.2 * rob_width
    right_middle = min_x + rob_width / 2 + 0.2 * rob_width


    for i, contour in enumerate(contours_raw):
        area = cv2.contourArea(contour)
        Xsum = 0
        Ysum = 0

        for j, point in enumerate(contour):
            Xsum += point[0][0]
            Ysum += point[0][1]
        # Calculate the centroid
        middle = [Xsum / (len(contour)), Ysum / (len(contour))]

        if area < 10:
            continue
        elif middle[0] < left_middle:
            continue
        elif middle[0] > right_middle:
            continue
        elif middle[1] > rob_y_mid:
            continue
        else:
            filtered_contours.append(contour)
            continue
    return filtered_contours

def facial_recognition(image):
    # Detect faces in the image
    face_locations = face_recognition.face_locations(image)
    # Check if any face is found
    if face_locations:
        return True
    else:
        return False

def get_circulaire_contour(contours, min_circle_length, max_circle_length, point_skip=3, ):
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
                if circle_length >= min_circle_length and circle_length <= max_circle_length:
                    list_curves.append(circle_length)
        if len(list_curves) > 0:
            circulaire_contours.append(contour)
    return circulaire_contours

def circle_long_enough(contour, i, point_skip, p1, v1, a1):
    circle_length = point_skip
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
        if a2 <= a1 and a2 >= a1 - 180 and check_vector_angle_difference(contour, i+k, 1, 3):
            circle_length += 1
        else:
            return circle_length
    return circle_length

def check_vector_angle_difference(contour, i, max_difference=1, point_skip=3):
    p1 = contour[(i) % len(contour)][0]
    p2 = contour[(i - point_skip) % len(contour)][0]
    p3 = contour[(i + point_skip) % len(contour)][0]

    v1 = get_vector(p2, p1)
    v2 = get_vector(p3, p1)

    a1 = get_angle(v1, v2)

    if a1 < max_difference:
        return False
    else: 
        return True

def get_vector(p1, p2):
    v = np.array(p1) - np.array(p2)
    return v

def get_angle(v1, v2):
    angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    angle = (angle ** 2) ** 0.5
    if angle > 180:
        angle = 360 - angle 
    return angle

def check_eyes(heads, potential_eyes, error=5):
    good_heads = []
    for head in heads:
        x_values = head[:, 0, 0]
        y_values = head[:, 0, 1]
        min_x = min(x_values) - error
        max_x = max(x_values) + error
        min_y = min(y_values) - error
        max_y = max(y_values) + error
        eyes = []
        for eye in potential_eyes:
            x_values = eye[:, 0, 0]
            y_values = eye[:, 0, 1]
            eye_min_x = min(x_values)
            eye_max_x = max(x_values)
            eye_min_y = min(y_values)
            eye_max_y = max(y_values)
            if eye_min_x > min_x and eye_max_x < max_x and eye_min_y > min_y and eye_max_y < max_y:
                eyes.append(eye)
            else:
                continue
        if len(eyes) >= 2:
            good_heads.append(head)
    return good_heads, eyes

def slice_contours(heads, point_skip=3):
    sliced_heads = []
    for head in heads:
        last_i = 0
        for i in range(len(head)):
            
            # get points from contour to get an angle
            p1 = head[i][0]
            p2 = head[(i + point_skip) % len(head)][0]
            p3 = head[(i - point_skip) % len(head)][0]
            # create vectors between the points (p1, p2) (p3, p2)
            v1 = get_vector(p1, p2)
            v2 = get_vector(p1, p3)
            # get angle between v1 and v2
            angle = get_angle(v1, v2)
            # check if angle is in boundaries
            if i == len(head)-1:
                sliced_heads.append(head[last_i:])
            elif angle < 200:
                continue
            else:
                if i-last_i > 10:
                    sliced_heads.append(head[last_i:i])
                last_i = i
    return sliced_heads

def merge_contours_with_tolerance(contours, tolerance=5):
    # Create a list to hold the y minimums
    y_mins = []

    # Iterate through each contour to find y minimum values
    for contour in contours:
        y_min = np.min(contour[:, :, 1])
        y_mins.append(y_min)

    # Sort the y minimum values and keep track of their original contours
    y_mins_sorted_indices = np.argsort(y_mins)
    sorted_y_mins = np.array(y_mins)[y_mins_sorted_indices]
    sorted_contours = [contours[i] for i in y_mins_sorted_indices]

    # List to hold the merged contours
    merged_contours = []
    current_group = []

    # Merge contours based on y minimum similarity within the tolerance
    for i in range(len(sorted_y_mins)):
        if not current_group:
            current_group.append(sorted_contours[i])  # Start a new group

        # Check if the current y_min is within the tolerance of the first y_min in the group
        elif abs(sorted_y_mins[i] - np.min([np.min(cnt[:, :, 1]) for cnt in current_group])) <= tolerance:
            current_group.append(sorted_contours[i])  # Add to the current group
        else:
            # Merge the current group and add to the result
            merged_contour = np.vstack(current_group)
            merged_contour = cv2.approxPolyDP(merged_contour, epsilon=1.0, closed=True)
            merged_contours.append(merged_contour)
            current_group = [sorted_contours[i]]  # Start a new group with the current contour

    # Handle the last group if it exists
    if current_group:
        merged_contour = np.vstack(current_group)
        merged_contour = cv2.approxPolyDP(merged_contour, epsilon=1.0, closed=True)
        merged_contours.append(merged_contour)

    return merged_contours

if __name__ == "__main__":
    main()