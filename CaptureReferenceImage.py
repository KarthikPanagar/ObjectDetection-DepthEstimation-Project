import cv2 as cv
import numpy as np
import time

# Setting parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# Colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv.FONT_HERSHEY_COMPLEX

# Reading class names from text file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Setting up camera
camera = cv.VideoCapture(0)
counter = 0
capture = False
number = 0

def ObjectDetector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(classes) > 0:
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            
            if isinstance(classid, np.ndarray):
                class_name = class_names[classid[0]]
            else:
                class_name = class_names[classid]
            
            label = "%s : %f" % (class_name, score)
            cv.rectangle(image, box, color, 2)
            cv.putText(image, label, (box[0], box[1]-10), fonts, 0.5, color, 2)

while True:
    ret, frame = camera.read()
    original = frame.copy()
    ObjectDetector(frame)
    cv.imshow('original', original)

    if capture == True and counter < 10:
        counter += 1
        cv.putText(frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
    else:
        counter = 0

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('c'):
        capture = True
        number += 1
        cv.imwrite(f'ReferenceImages/image{number}.png', original)
    if key == ord('q'):
        break

cv.destroyAllWindows()
