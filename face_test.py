import face_recognition
import os
import cv2
import numpy as np

KNOWN_FACES_DIR = "database"
TEST_IMAGE_PATH = "test1.jpg"  # changed to the image that exists in directory

known_encodings = []
known_names = []

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces
for file in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, file)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(os.path.splitext(file)[0])

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Error: Could not find {TEST_IMAGE_PATH}")
    exit()

# Load test image
test_image = face_recognition.load_image_file(TEST_IMAGE_PATH)
test_encodings = face_recognition.face_encodings(test_image)
face_locations = face_recognition.face_locations(test_image)

image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(face_locations, test_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "Unknown"

    if len(known_encodings) > 0:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(image_bgr, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

cv2.imshow("Result", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
