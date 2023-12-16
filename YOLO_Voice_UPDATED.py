#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyttsx3')
get_ipython().system('pip install --upgrade pyttsx3')


# In[ ]:


get_ipython().system('pip install ultralytics')


# In[ ]:


get_ipython().system('pip install gtts')


# In[ ]:


''''from ultralytics import YOLO
import cv2
import math
from gtts import gTTS
import os

# Function to convert text to speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # This command plays the mp3 file, make sure to install mpg321

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "rock", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detected_objects = []

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print("Class name -->", class_name)

            detected_objects.append(class_name)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

    # Convert detected objects to a sentence
    if detected_objects:
        output_text = "Detected objects: " + ', '.join(detected_objects)
        print(output_text)
        speak(output_text)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''


# In[ ]:


import cv2
from gtts import gTTS
import pyttsx3
from ultralytics import YOLO
import objc
import os
import time

# Function to convert text to speech using pyttsx3

def speak(text):
    os.system(f"say {text}")
    time.sleep(15)

'''def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
'''
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "rock", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detected_objects = []

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # confidence
            confidence = round(box.conf[0].item(), 2)
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print("Class name -->", class_name)

            detected_objects.append(class_name)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # object details
            org = (x1, y1 - 10)  # Adjusting the text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, f"{class_name} {confidence}", org, font, fontScale, color, thickness)

    # Convert detected objects to a sentence
    if detected_objects:
        output_text = "Detected objects: " + ', '.join(detected_objects)
        print(output_text)
        speak(output_text)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


pip install --upgrade pyttsx3


# In[1]:


get_ipython().system('pip install picamera')


# In[ ]:


import cv2
from gtts import gTTS
import pyttsx3
from ultralytics import YOLO
import objc
import os
import time
from picamera import PiCamera
from picamera.array import PiRGBArray

# Function to convert text to speech using pyttsx3
def speak(text):
    os.system(f"say {text}")
    time.sleep(15)

# start the Raspberry Pi camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "rock", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    img = frame.array
    results = model(img, stream=True)

    detected_objects = []

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # confidence
            confidence = round(box.conf[0].item(), 2)
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print("Class name -->", class_name)

            detected_objects.append(class_name)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # object details
            org = (x1, y1 - 10)  # Adjusting the text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, f"{class_name} {confidence}", org, font, fontScale, color, thickness)

    # Convert detected objects to a sentence
    if detected_objects:
        output_text = "Detected objects: " + ', '.join(detected_objects)
        print(output_text)
        speak(output_text)

    cv2.imshow('Raspberry Pi Camera', img)
    raw_capture.truncate(0)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

