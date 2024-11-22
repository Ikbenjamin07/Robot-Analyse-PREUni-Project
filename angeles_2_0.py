import cv2
import numpy as np
import os 
import robot_data as robot_data
import random


def main():
    angle()

def angle():
    folder_path = 'img/'
    score = []

    for filename in os.listdir(folder_path):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        image = load_image(file_path)
        print(file_path)
        contours_raw = get_contours(image)
        contours_raw = remove_double(contours_raw)
        angle_points = get_angulair_contours(contours_raw)

        # length = int(get_total_length_of_contours(circulaire_contours))
        # length_raw = int(get_total_length_of_contours(contours_raw))
        # area = int(get_total_opp_of_contours(circulaire_contours))

    #     score.append(length/length_raw)
    # return score
        # Maak een output afbeelding
        #output_image = image.copy()
#
        #for coord in angle_points:
        #    coord = coord[0]
        #    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #    cv2.polylines(output_image, [contour], isClosed=False, color=color, thickness=2)
#
        ## Toon de originele afbeelding en de afbeelding met gedetecteerde contouren
        #cv2.imshow(f'Circle Image: {filename}', output_image)
        score.append(len(angle_points))
    # Wacht tot een toets is ingedrukt om alle vensters te sluiten
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(score)
    return score

def remove_double(contours):
    print('KABOOM')
    new_contours = []
    for contour in contours: 
        if cv2.contourArea(contour) < 5:
            new_contour = contour[:round(len(contour)/2)]
            new_contours.append(new_contour)
        else:
            new_contours.append(contour)
    return new_contours

def get_angulair_contours(contours):
    all_angle_points = []
    for contour in contours:
        #gets al the points in the contours that have a sharp enough angle
        # [i, i, i, i, i, i]
        angle_points = get_angle_point(contour)
        for angle_point in angle_points:
            all_angle_points.append(contour[angle_point[0]])
            # # gets the length of the angle from [<-, i, ->]
            # # [neg_length, pos_length]
            # angle_length_neg, angle_length_pos = get_angle_length(angle_point, contour)
            # # gets the part of the contour that has an angle an makes a new contour from it 
            # # [[x,y][x,y] enz...]
            # angle_contour = get_contour_fragment(angle_point, angle_length_neg, angle_length_pos)
            # angulair_contours.append(angle_contour)
    return all_angle_points





def get_angle_point(contour, point_skip=3, max_angle=55):
    angle_points = []
    for i, point in enumerate(contour): 
        try:
            point_A = contour[i-point_skip][0]
            point_B = contour[i][0]
            point_C = contour[i+point_skip][0]
        except:
            continue

        v_AB = get_vector(point_A, point_B)
        v_BC = get_vector(point_B, point_C)

        a_ABC = get_angle(v_AB, v_BC)

        if a_ABC < max_angle and a_ABC != 0:
            angle_points.append([i, a_ABC])
    return angle_points


def get_angle_length(angle_point, contour, point_skip=3, max_angle_devation=5):
    i, angle = angle_point
    start_point = i
    # pos_length
    try:
        point_A = contour[start_point-point_skip][0]
        point_B = contour[start_point][0]
        point_C = contour[start_point+point_skip][0]
    except:
        return

    v_AB = get_vector(point_A, point_B)
    v_BC = get_vector(point_B, point_C)

    a_ABC = get_angle(v_AB, v_BC)

    while i <= len(contour):
        break
    return 

#def get_contour_fragment(angle_point, angle_length_neg, angle_length_pos):
#    ...

#image processing functions
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
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detecteer randen met de Canny-randdetectiemethode
    edges = cv2.Canny(blurred, 50, 150)

    # Vind de contouren in de afbeelding
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

#math functions
def get_vector(p1, p2):
    v = np.array(p1) - np.array(p2)
    return v

def get_angle(v1, v2):
    angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))

    angle = (angle ** 2) ** 0.5

    if angle > 180:
        angle = 360 - angle 

    return angle



# contour functios 

def make_shape_contour_drawable(contours):
    drawable_contours = []
    for contour in contours:
        drawable_contours.append(np.array([contour], dtype=np.int32))
    return drawable_contours

def get_total_length_of_contours(contours):
    total_length = 0
    for contour in contours:
        length = cv2.arcLength(contour, True)
        total_length = total_length + length
    return total_length

def get_total_opp_of_contours(contours):
    total_area = 0
    for contour in contours:
        total_area = total_area + cv2.contourArea(contour)
    return total_area


if __name__ == "__main__":
    main()