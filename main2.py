import os
import datetime
import time

import cv2
from pyzbar.pyzbar import decode
import numpy as np

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the whitelist and log files
whitelist_path = './whitelist.txt'
log_path = './log.txt'

# Read authorized users from whitelist file
with open(whitelist_path, 'r') as f:
    authorized_users = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

# Video capture object
cap = cv2.VideoCapture(0)

most_recent_access = {}
time_between_logs_th = 5

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and write "Face Detected"
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Student Spotted', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # QR Code scanning
    qr_info = decode(frame)
    if qr_info is not None and len(qr_info) > 0:
        qr = qr_info[0]
        data = qr.data.decode()
        rect = qr.rect
        polygon = qr.polygon

        if data in authorized_users:
            cv2.putText(frame, 'ACCESS GRANTED', (rect.left, rect.top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if data not in most_recent_access or time.time() - most_recent_access[data] > time_between_logs_th:
                most_recent_access[data] = time.time()
                with open(log_path, 'a') as f:
                    f.write('{},{}\n'.format(data, datetime.datetime.now()))

        else:
            cv2.putText(frame, 'ACCESS DENIED', (rect.left, rect.top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        frame = cv2.rectangle(frame, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height),
                              (0, 255, 0), 5)
        frame = cv2.polylines(frame, [np.array(polygon)], True, (255, 0, 0), 5)

    # Display the frame
    cv2.imshow('webcam', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

