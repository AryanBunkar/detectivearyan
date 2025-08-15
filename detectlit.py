# import streamlit as st
# import os
# import pandas as pd
# from datetime import datetime

# st.title("Attendence data - ")
# csv_filename = datetime.now().strftime("%Y-%m-%d") + ".csv"
# if os.path.exists(csv_filename)and os.path.getsize(csv_filename) > 0:
#     df = pd.read_csv(csv_filename)
#     st.dataframe(df)
# else:
#     st.warning("No Attendence file found yet...")
    
# import numpy as np
# import face_recognition
# import cv2
# from datetime import datetime
# import csv
# import pandas as pd
# import streamlit as st
# import os
# import time

# st.set_page_config(page_title="Face Detection & Attendance System", layout="wide")
# st.title("📸 Face Detection & Attendance System")

# # Load known faces directly
# known_face_encodings = [
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\aryan.jpg"))[0],
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\harshit.jpg"))[0],
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\tanmay.jpg"))[0],
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\pranav.jpg"))[0],
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\tanvi.jpg"))[0],
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\vaibhavi.jpg"))[0],
#     face_recognition.face_encodings(face_recognition.load_image_file(
#         r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\mritunjai.jpeg"))[0],
# ]

# known_faces_names = ["aryan", "harshit", "tanmay", "pranav", "tanvi", "vaibhavi", "mritunjai"]
# students = known_faces_names.copy()

# # CSV filename
# current_date = datetime.now().strftime("%Y-%m-%d")
# csv_filename = f"{current_date}.csv"

# # Create CSV if not exists
# if not os.path.exists(csv_filename):
#     with open(csv_filename, "w", newline="") as f:
#         csv.writer(f).writerow(["Name", "Time"])

# # Camera start/stop buttons
# start_camera = st.button("▶ Start Camera")
# stop_camera = st.button("⏹ Stop Camera")

# frame_placeholder = st.empty()
# table_placeholder = st.empty()

# if start_camera:
#     video_capture = cv2.VideoCapture(0)

#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         face_names = []

#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distance)

#             if matches[best_match_index]:
#                 name = known_faces_names[best_match_index]
#                 face_names.append(name)

#                 if name in students:
#                     students.remove(name)
#                     current_time = datetime.now().strftime("%H:%M:%S")
#                     with open(csv_filename, "a", newline="") as f:
#                         csv.writer(f).writerow([name, current_time])
#             else:
#                 face_names.append("Unknown")

#         # Draw rectangles and names
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         frame_placeholder.image(frame, channels="BGR")

#         # Show updated attendance table
#         if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0:
#             df = pd.read_csv(csv_filename)
#             table_placeholder.dataframe(df)

#         time.sleep(0.05)

#         # Stop when stop_camera is clicked
#         if stop_camera:
#             break


#     video_capture.release()
#     cv2.destroyAllWindows()
# # Show CSV data after camera stops
# df = pd.read_csv(csv_filename)
# st.dataframe(df)


import numpy as np
import cv2
import pandas as pd
import streamlit as st
import os
from datetime import datetime
import time

st.set_page_config(page_title="Face Detection & Attendance System", layout="wide")
st.title("📸 Face Detection & Attendance System (OpenCV Only)")

# Path to Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Names list (must match trained IDs)
known_faces_names = ["aryan", "harshit", "tanmay", "pranav", "tanvi", "vaibhavi", "mritunjai"]
students = known_faces_names.copy()

# Train face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
training_data = []
labels = []

# Folder where training images are stored
training_dir = r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo"

for idx, name in enumerate(known_faces_names):
    img_path = os.path.join(training_dir, f"{name}.jpg")  # ensure all photos have same name as list
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        training_data.append(img)
        labels.append(idx)

recognizer.train(training_data, np.array(labels))

# CSV filename
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"{current_date}.csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, "w") as f:
        f.write("Name,Time\n")

start_camera = st.button("▶ Start Camera")
stop_camera = st.button("⏹ Stop Camera")

frame_placeholder = st.empty()
table_placeholder = st.empty()

if start_camera:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 80:
                name = known_faces_names[face_id]
                color = (0, 255, 0)
                if name in students:
                    students.remove(name)
                    with open(csv_filename, "a") as f:
                        f.write(f"{name},{datetime.now().strftime('%H:%M:%S')}\n")
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_placeholder.image(frame, channels="BGR")

        if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0:
            df = pd.read_csv(csv_filename)
            table_placeholder.dataframe(df)

        if stop_camera:
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
    st.dataframe(df)
