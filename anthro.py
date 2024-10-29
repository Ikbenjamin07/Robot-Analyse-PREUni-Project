import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

def main():
    #faces = facial()
    gezicht()

def facial():
    # Define the path to the folder
    folder_path = 'img/'
    face_similarity_scores = []
    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        # Load the image
        file_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(file_path)  # Replace with your image path

        # Detect faces in the image
        face_locations = face_recognition.face_locations(image)
        # Check if any face is found
        if face_locations:
            # For the first detected face (assuming only one face in the image)
            top, right, bottom, left = face_locations[0]

            # Calculate the similarity as the proportion of the image taken by the face
            image_area = image.shape[0] * image.shape[1]
            face_area = (bottom - top) * (right - left)
            face_similarity_score = (face_area / image_area) * 100  # As a percentage

            # Draw a rectangle around the face
            image_with_face = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_with_face, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the similarity score and the image
            print(f"Face detected with similarity score: {face_similarity_score:.2f}%")
            cv2.imshow("Detected Face", image_with_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            face_similarity_scores.append(face_similarity_score)
        else:
            print("No face detected in the image.")
            face_similarity_scores.append(0)
    return face_similarity_scores







if __name__ == "__main__":
    main()

