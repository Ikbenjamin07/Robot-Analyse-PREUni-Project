import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

def main():
    face()

def face():
    folder_path = 'img/'
    scores = []

    for filename in os.listdir(folder_path):
        print(filename)
        score = 0
        file_path = os.path.join(folder_path, filename)
        image = load_image(file_path)
        contours_raw = get_contours(image)

        filtered_contours = filter(contours_raw, contours_raw)
        potential_heads = get_circulaire_contour(filtered_contours, 50, 999)
        potential_eyes = get_circulaire_contour(filtered_contours, 5, 50)
        sliced_heads = slice_contours(potential_heads)
        sliced_heads = filter(contours_raw, sliced_heads)
        merged_heads = merge_contours_by_proximity(sliced_heads)
        if merged_heads:
            score = check_eyes(merged_heads, potential_eyes)
        else:
            score = 0
        face_recognised = facial_recognition(image)
        if face_recognised:
            score += 20
        scores.append(score / 20)
    return scores
    
    

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

def filter(contours_raw, contours):
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


    for i, contour in enumerate(contours):
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

def check_eyes(heads, potential_eyes, error=0):
    score = 0
    checked_eye_height = False
    checked_eye_sides = False
    checked_eye_sizes = False
    for i, head in enumerate(heads):

        x_values = head[:, 0, 0]
        y_values = head[:, 0, 1] 
        min_x = min(x_values) - error
        max_x = max(x_values) + error
        min_y = min(y_values) - error
        max_y = max(y_values) + error
        eyes = []
        eye_middles = []
        eye_sizes = []
        for eye in potential_eyes:
            x_values = eye[:, 0, 0]
            y_values = eye[:, 0, 1]
            eye_min_x = min(x_values)
            eye_max_x = max(x_values)
            eye_min_y = min(y_values)
            eye_max_y = max(y_values)

            Xsum = 0
            Ysum = 0
            for j, point in enumerate(eye):
                Xsum += point[0][0]
                Ysum += point[0][1]
            # Calculate the centroid
            middle = [Xsum / (len(eye)), Ysum / (len(eye))]
            size = cv2.contourArea(eye)

            if eye_min_x > min_x and eye_max_x < max_x and eye_min_y > min_y and eye_max_y < max_y:
                eye_middles.append(middle)
                eyes.append(eye)
                eye_sizes.append(size)
            else:
                continue

        if len(eyes) == 2 or len(eyes) == 3:
            eyes_same_height = False
            for i in range(len(eye_middles)):
                for j in range(i + 1, len(eye_middles)):
                    if abs(eye_middles[i][1] - eye_middles[j][1]) <= 5:
                        eyes_same_height = True
                        break  # Break out of the `j` loop
                if eyes_same_height:
                    break  # Break out of the `i` loop if condition was met

            eyes_same_size = False
            for i in range(len(eye_sizes)):
                for j in range(i + 1, len(eye_sizes)):
                    if abs(eye_sizes[i] - eye_sizes[j]) <= 20:
                        eyes_same_size = True
                        break  # Break out of the `j` loop
                if eyes_same_size:
                    break  # Break out of the `i` loop if condition was met

            Xsum = 0
            Ysum = 0
            for j, point in enumerate(head):
                Xsum += point[0][0]
                Ysum += point[0][1]
            middle = [Xsum / (len(head)), Ysum / (len(head))]

            above_count = 0
            below_count = 0
            #Count how many y-values are above and below the target_value
            for eye in eye_middles:
                x_value = eye[0]
                if x_value > middle[0]:
                    above_count += 1
                elif x_value < middle[0]:
                    below_count += 1
            eyes_opposite_sides = above_count > 0 and below_count > 0
            
            if eyes_same_height and checked_eye_height == False:
                checked_eye_height = True
                score += 1
            if eyes_opposite_sides and checked_eye_sides == False:
                checked_eye_sides = True
                score += 4
            if eyes_same_size and checked_eye_sizes == False:
                checked_eye_sizes = True
                score += 1
    return score

def slice_contours(heads, point_skip=5):
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
            elif angle < 200 and angle > 75 :
                continue
            else:
                if i - last_i > 10:
                    sliced_heads.append(head[last_i:i])
                    last_i = i
    return sliced_heads

def merge_contours_by_proximity(contours, tolerance=5):
    # List to hold the merged contours
    merged_contours = []
    current_group = []

    # Function to get the first and last points of a contour
    def first_last_points(contour):
        return contour[0][0], contour[-1][0]  # Assumes contour is in (x, y) format with an extra dimension

    # Iterate over each contour to group and merge them based on proximity of endpoints
    for contour in contours:
        if not current_group:
            current_group.append(contour)  # Start a new group if it's empty
            continue

        # Check if the first or last point of the current contour is close to any endpoint in the current group
        endpoints_in_group = [first_last_points(cnt) for cnt in current_group]
        current_first, current_last = first_last_points(contour)

        # Check proximity of endpoints (first or last point) within the group
        within_tolerance = any(
            np.linalg.norm(np.array(current_first) - np.array(pt)) <= tolerance or
            np.linalg.norm(np.array(current_last) - np.array(pt)) <= tolerance
            for endpoints in endpoints_in_group for pt in endpoints
        )
        
        if within_tolerance:
            current_group.append(contour)  # Add to the current group
        else:
            # Merge the current group if not empty
            if current_group:
                merged_contour = np.vstack(current_group)
                # Check if merged_contour is valid before passing to approxPolyDP
                if merged_contour.size > 0:
                    merged_contour = cv2.approxPolyDP(merged_contour, epsilon=1.0, closed=True)
                    if merged_contour is not None:
                        merged_contours.append(merged_contour)
                    else:
                        print("Warning: approxPolyDP returned None for a contour.")
                else:
                    print("Warning: Attempted to merge an empty contour group.")
            current_group = [contour]  # Start a new group with the current contour

    # Handle the last group if it exists
    if current_group:
        merged_contour = np.vstack(current_group)
        # Check if merged_contour is valid before passing to approxPolyDP
        if merged_contour.size > 0:
            merged_contour = cv2.approxPolyDP(merged_contour, epsilon=1.0, closed=True)
            if merged_contour is not None:
                merged_contours.append(merged_contour)
            else:
                print("Warning: approxPolyDP returned None for the last contour.")
        else:
            print("Warning: Last group is empty; nothing to merge.")

    return merged_contours if merged_contours else None  # Return None if no valid contours



if __name__ == "__main__":
    main()