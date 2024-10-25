import cv2
import face_recognition

# Load the image
image = face_recognition.load_image_file("img/Miko2.jpg")  # Replace with your image path

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
else:
    print("No face detected in the image.")
