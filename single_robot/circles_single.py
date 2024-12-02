import cv2
import numpy as np
import random
import os

def main():
    circle()

def circle():
    folder_path = 'single_robot/robots/'
    score = []
    sorted_files = sort_files()
    for filename in sorted_files:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        image = load_image(file_path)
        print(file_path)
        contours_raw = get_contours(image)
        circulaire_contours = get_circulaire_contour(contours_raw)

        length = int(get_total_length_of_contours(circulaire_contours))
        length_raw = int(get_total_length_of_contours(contours_raw))
        area = int(get_total_opp_of_contours(circulaire_contours))

        score.append(len(circulaire_contours))
    return score
    #     # Maak een output afbeelding
    #     output_image = image.copy()

    #     for contour in circulaire_contours:
    #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #         cv2.polylines(output_image, [contour], isClosed=False, color=color, thickness=2)

    #     # Toon de originele afbeelding en de afbeelding met gedetecteerde contouren
    #     cv2.imshow(f'Circle Image: {name}', output_image)

    # # Wacht tot een toets is ingedrukt om alle vensters te sluiten
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

def check_vector_angle_difference(contour, i, max_difference=1, point_skip=3):
    try:
        p1 = contour[(i)][0]
        p2 = contour[(i - point_skip)][0]
        p3 = contour[(i + point_skip)][0]
    except:
        return False

    v1 = get_vector(p2, p1)
    v2 = get_vector(p3, p1)

    a1 = get_angle(v1, v2)

    if a1 < max_difference:
        return False
    else: 
        return True 


#circle functions
def get_circulaire_contour(contours, point_skip=3):
    shape_contours = []
    holding_list_of_contours = []
    p = 0
    q = 0
    r = 0
    s = 0
    # print(len(contours))
    for contour in contours:
        list_curves = []
        # check if there are circulair ish shapes in the contour 
        for i in range(len(contour)):
            circle_length = circle_long_enough(contour, i, point_skip)
            if circle_length >= 5:
                list_curves.append([circle_length, i])

        #  get the curved parts of the contours 
        if len(list_curves) != 0:
            list_curves_clean = filter_circles(list_curves)
            shape_contour = calculate_contour_circle(contour, list_curves_clean)
            for contour in shape_contour:
                shape_contours.append(contour)

    # get rid of angels
    for contour in shape_contours:
        _, _, curvature = big_enough_curvature(contour, point_skip=3)
        contour = check_angle(contour, curvature, max_angle=1)
        for sub_contour in contour:
            if len(sub_contour) > 20:
                q += 1
                holding_list_of_contours.append(sub_contour)

    shape_contours = holding_list_of_contours
    holding_list_of_contours = []

    # get rid of angels
    for contour in shape_contours:
        _, _, curvature = big_enough_curvature(contour, point_skip=3)
        contour = check_angle(contour, curvature, max_angle=1)
        for sub_contour in contour:
            if len(sub_contour) > 20:
                q += 1
                holding_list_of_contours.append(sub_contour)

    shape_contours = holding_list_of_contours
    holding_list_of_contours = []

    # get rid of angels
    for contour in shape_contours:
        _, _, curvature = big_enough_curvature(contour, point_skip=1)
        contour = check_angle(contour, curvature, max_angle=1)
        for sub_contour in contour:
            if len(sub_contour) > 20:
                q += 1
                holding_list_of_contours.append(sub_contour)

    shape_contours = holding_list_of_contours
    holding_list_of_contours = []

    #hough lines
    for contour in shape_contours:
        lines = detect_lines_from_contour(contour)
        # Verwijder punten dicht bij de gedetecteerde lijnen
        filtered_contour = remove_points_near_lines(contour, lines)
        for contour in filtered_contour:
            if len(contour) > 5:
                holding_list_of_contours.append(contour)
    
    shape_contours = holding_list_of_contours
    holding_list_of_contours = []

    # # check for streight lines again...
    # for contour in shape_contours:
    #     contour = line_check2(contour)
    #     if len(contour) > 20:
    #         s += 1
    #         holding_list_of_contours.append(contour)
    
    # shape_contours = holding_list_of_contours
    # holding_list_of_contours = []

    
    # get rid of streight lines 
    for contour in shape_contours:
        _, _, curvature = big_enough_curvature(contour, point_skip=1)
        contour = line_check(contour, curvature)
        for sub_contour in contour:
            if len(sub_contour) > 20:
                p += 1
                holding_list_of_contours.append(sub_contour)

    shape_contours = holding_list_of_contours
    holding_list_of_contours = []

    # # get rid of streight lines 
    # for contour in shape_contours:
    #     _, _, curvature = big_enough_curvature(contour, point_skip=1)
    #     contour = filter_contour_by_curvature(contour, curvature)
    #     for sub_contour in contour:
    #         if len(sub_contour) > 20:
    #             p += 1
    #             holding_list_of_contours.append(sub_contour)

    # shape_contours = holding_list_of_contours
    # holding_list_of_contours = []


    # for contour in shape_contours:
    #     _, _, curvature = big_enough_curvature(contour)
    #     print(curvature)

    # print(p, q, r, s)
    drawable_contour = make_shape_contour_drawable(shape_contours)
    return drawable_contour

def circle_long_enough(contour, i, point_skip):

    circle_length = point_skip

    try:
        p1 = contour[i][0]
        p2 = contour[(i + point_skip)][0] 
    except:
        return circle_length

    v1 = np.array(p2) - np.array(p1)

    a1 = np.degrees(np.arctan2(v1[1], v1[0]))


    for k in range(point_skip + 1, len(contour)): 
        try:
            p3 = contour[(i + k)][0]
        except: 
            break
        v2 = get_vector(p1, p3)
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
        
        

        if a2 <= a1 and a2 >= a1 - 180 and check_vector_angle_difference(contour, i+k) == True:
            circle_length += 1
        else:
            return circle_length
    return circle_length

def big_enough_curvature(contour, point_skip=1):
    contour = np.array(contour)
    n = contour.shape[0]
    curvature = np.zeros(n)

    for i in range(point_skip, n - point_skip):
        # Haal de coördinaten van de punten
        x_prev, y_prev = contour[i - point_skip]
        x_curr, y_curr = contour[i]
        x_next, y_next = contour[i + point_skip]

        # Bereken de richtingsvectoren
        T_prev = np.array([x_curr - x_prev, y_curr - y_prev], dtype=float)
        T_next = np.array([x_next - x_curr, y_next - y_curr], dtype=float)

        # Normalizeer de richtingsvectoren
        if np.linalg.norm(T_prev) > 0 and np.linalg.norm(T_next) > 0:
            T_prev /= np.linalg.norm(T_prev)
            T_next /= np.linalg.norm(T_next)

            # Bereken de kromming (verandering van richting)
            curvature[i] = np.linalg.norm(T_next - T_prev)

    # Bereken de gemiddelde kromming (exclusief de eerste en laatste punten)

    total_curvature = np.sum(curvature[1:-1])
    if total_curvature > 0:
        mean_curvature = np.mean(curvature[1:-1])
    else: 
        mean_curvature = 0

    return mean_curvature, total_curvature, curvature

def filter_circles(list_curves):
    list_to_remove_raw = []
    for i in range(len(list_curves)):
        lengte, punt = list_curves[i]
        for j in range(len(list_curves)):
            if i < j:
                andere_lengte, andere_punt = list_curves[j]
                if punt + lengte > andere_punt:
                    # Verwijder het punt met de kleinste lengte
                    if lengte > andere_lengte:
                        list_to_remove_raw.append(list_curves[j])
                    if lengte < andere_lengte:
                        list_to_remove_raw.append(list_curves[i])

    list_to_remove_clean = []
    seen = set()

    for item in list_to_remove_raw:
        tuple_item = tuple(item)  # Convert list to tuple
        if tuple_item not in seen:
            seen.add(tuple_item)
            list_to_remove_clean.append(item)


    for item in list_to_remove_clean:
        list_curves.remove(item)
    return list_curves

def get_surface_circle():
    ...

def calculate_contour_circle(contour, list_curves):
    shape_contour = []
    for item in list_curves:
        i = item[1]
        length = item[0]
        c = []
        for j in range(length):
            c.append(contour[(i+j)][0])
        c = np.array(c)
        shape_contour.append(c)
    return shape_contour

def line_check(contour, curvature):
    result = []
    count = 0
    start_index = 0
    for i, num in enumerate(curvature):
        if num < 0.1:
            count += 1
        else:
            if count > 10:
                # Split the list and remove the zeros
                result.append(contour[start_index:i - count])  # Before the zeros
                start_index = i
            count = 0  # Reset the count for non-zero

    # Handle the case for the last segment
    if count > 10:
        result.append(contour[start_index:len(contour) - count])  # Before the zeros
    else:
        result.append(contour[start_index:])  # Remaining part of the list

    return result

def line_check2(contour):
    list_of_points_to_delete = []
    for i in range(len(contour)):
        count = 0
        try:
            p1 = contour[i]
            p2 = contour[i+1]
        except:
            continue
        v1 = get_vector(p2, p1)
        for j in range(i + 1, len(contour)):
            p3 = contour[j]
            v2 = get_vector(p3, p1)
            if v1.all() == v2.all():
                count += 1
            else: 
                if count > int(len(contour)/1.125):
                    for k in range(j-count, j):
                        list_of_points_to_delete.append(contour[k].tolist())
                count = 0

        if count > int(len(contour)/1.125):
            for j in range(len(contour)-count, len(contour)):
                list_of_points_to_delete.append(contour[j].tolist())
    
    points_set = set(map(tuple, list_of_points_to_delete))
    
    # Filter de contour om alleen de punten die niet verwijderd moeten worden te behouden
    filtered_contour = [point for point in contour if tuple(point) not in points_set]
    
    return filtered_contour
    
def radius_check(contour):

    #get biggest differance between points
    max_distance = 0
    p1 = None
    p2 = None
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            distance = ((contour[j][0]-contour[i][0])**2+(contour[j][1]-contour[i][1])**2)**0.5
            if distance > max_distance:
                max_distance = distance
                p1, p2 = contour[i], contour[j]



    # get the biggest radiuses and the differance between those 
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

    if (x2-x1) == 0:
        a = 0
    else:
        a = (y2-y1)/(x2-x1)
    b = -1
    c = y1-(a*x1)

    max_radius2 = 0 
    for i in range(1, len(contour)-1):
        p = contour[i]
        radius2 = (a*p[0]+b*p[1]+c)/((a)**2+(b)**2)**0.5
        if radius2 > max_radius2:
            max_radius2 = radius2 

    radius1 = (((x2-x1)**2+(y2-y1)**2)**0.5)/2

    radius_differance = max_radius2/radius1

    return radius_differance

def check_angle(contour, curvature, max_angle=1):
    result = []
    count = 0
    start_index = 0

    for i, num in enumerate(curvature):
        if num >= max_angle:
            count += 1
        else:
            if count > 0:
                # Split the list and remove the zeros
                result.append(contour[start_index:i - count])  # Before the zeros
                start_index = i
            count = 0  # Reset the count for non-zero

    # Handle the case for the last segment
    if count > 0:
        result.append(contour[start_index:len(contour) - count])  # Before the zeros
    else:
        result.append(contour[start_index:])  # Remaining part of the list
    return result

def detect_lines_from_contour(contour, threshold=20):
    # Maak een lege afbeelding die groot genoeg is om de contour te bevatten
    img = np.zeros((500, 500), dtype=np.uint8)  # Pas de grootte aan indien nodig

    contour = np.array(contour, dtype=np.int32)


    # Teken de contour op de lege afbeelding
    cv2.drawContours(img, [contour], -1, 255, thickness=1)

    # Voer edge detectie uit met Canny
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    if len(contour) < threshold*2:
        threshold = int((len(contour)/2))
    else:
        if radius_check(contour) > 0.5:
            threshold = int((len(contour)/4))
        else:
            threshold = threshold

    # Voer de Hough-transformatie uit om lijnen te detecteren
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=threshold)

    # cv2.imshow("a", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    detected_lines = []
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Bereken de start- en eindpunten van de lijn
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            detected_lines.append(((x1, y1), (x2, y2)))
    return detected_lines

def point_line_distance(point, line):
    """Bereken de afstand van een punt tot een lijn."""
    (x0, y0) = point
    (x1, y1), (x2, y2) = line
    
    # Gebruik de formule voor de afstand van een punt tot een lijn
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    if denominator == 0:
        return float('inf')  # Vermijd deling door nul
    return numerator / denominator

def remove_points_near_lines(contour, lines, distance_threshold=3):
    """Verwijder punten dicht bij lijnen en maak kleine contouren aan."""
    filtered_contours = []
    current_contour = []

    if len(lines) == 0:
        filtered_contours.append(contour)
        return filtered_contours
    
    for point in contour:
        x, y = point[0], point[1]
        keep_point = True
        
        for line in lines:
            distance = point_line_distance((x, y), line)
            if distance < distance_threshold:
                keep_point = False
                break
        
        if keep_point:
            current_contour.append(point)
        else:
            # Als het huidige contour niet leeg is, voeg het toe aan de lijst
            if current_contour:
                filtered_contours.append(np.array(current_contour))
                current_contour = []  # Reset de huidige contour

    # Voeg de laatste contour toe als deze niet leeg is
    if current_contour:
        filtered_contours.append(current_contour)

    return filtered_contours

def filter_contour_by_curvature(contour, curvature):
    # Zet de contour om in een lijst voor eenvoudiger manipulatie
    filtered_contour = []
    contour_holder = []
    n = curvature.shape[0]
    i = 0

    while i < n:
        # Zoek naar een segment van 10 punten
        if i + 10 <= n:
            segment = curvature[i:i + 10]
            # Tel het aantal nullen in het segment
            zero_count = np.sum(segment == 0)
            
            if zero_count > 6:
                # Verwijder het segment (overslaan in de filtered_contour)
                filtered_contour.append(contour_holder)
                contour_holder= []
                i += 10  # Sla dit segment over
            else:
                # Voeg het punt toe aan de nieuwe contour
                contour_holder.append(contour[i])
                i += 1
        else:
            # Voeg de resterende punten toe aan de nieuwe contour
            filtered_contour.extend(contour[i:n])
            break

    return filtered_contour


#contour functions


def sort_files():
    # Path to the directory containing images
    directory = "single_robot/robots/"

    # List all files in the directory
    files = os.listdir(directory)

    # Filter out non-image files if needed (e.g., .jpg, .png, .jpeg)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files alphabetically
    sorted_files = sorted(image_files)

    return sorted_files



def calculate_angulair_contour(contour, i , length):
    shape_contour = []
    for j in range(length):
        shape_contour.append(contour[(i+j)][0])
        shape_contour.append(contour[(i-j)][0])
    shape_contour = np.array(shape_contour)
    print(cv2.contourArea(shape_contour))
    return shape_contour

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