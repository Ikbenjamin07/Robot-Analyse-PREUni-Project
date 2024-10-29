import cv2

def main():
    image = load_image()
    inverted_image = cv2.bitwise_not(image)
    contours_raw = get_contours(inverted_image)

    filtered_contours = []
    width, height = image.shape[:2]
    for i, contour in enumerate(contours_raw):
        area = cv2.contourArea(contour)
        Xsum = 0
        Ysum = 0

        for j, point in enumerate(contour):
            Xsum += point[0][0]
            Ysum += point[0][1]
        # Calculate the centroid
        middle = [Xsum / (len(contour)), Ysum / (len(contour))]
        left_middle = width / 2
        right_middle = width / 2 + width * 0.1

        #print(Xsum)
        #print(len(contour))
        #print(middle[0])
        #print(' -------- ')

        if area < 10:
            continue
        elif middle[0] < left_middle:
            print('excluded')
            continue
        elif middle[0] > right_middle:
            continue
        else:
            filtered_contours.append(contour)



    contoured_image = cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', contoured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def load_image():
    # Laad de afbeelding
    image = cv2.imread('img/nao.jpg')
    if image is None:
        print("Afbeelding niet gevonden!")
        exit()
    return image


def get_contours(image):
    # Converteer de afbeelding naar grijswaarden
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gebruik een Gaussiaanse vervaging om ruis te verminderen
    #blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Detecteer randen met de Canny-randdetectiemethode
    edges = cv2.Canny(gray, 50, 150)

    # Vind de contouren in de afbeelding
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


if __name__ == "__main__":
    main()