import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 19 #INCHES
CAR_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    
# setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector function /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        
        if isinstance(classid, np.ndarray):
            class_name = class_names[classid[0]]
        else:
            class_name = class_names[classid]
        
        label = "%s : %f" % (class_name, score)

        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        width = box[2] - box[0]
        height = box[3] - box[1]
        diagonal = np.sqrt(width**2 + height**2)  # Calculating diagonal length
        
        if class_name == 'car':
            data_list.append([class_name, width, height, diagonal, (box[0], box[1]-2)])
        elif class_name == 'cell phone':
            data_list.append([class_name, width, height, diagonal, (box[0], box[1]-2)])
    
    return data_list


def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# Reading the reference image from dir 
ref_car = cv.imread('ReferenceImages/15c.jpg')
ref_mobile = cv.imread('ReferenceImages/image7.png')

# Getting the object data from the reference images
mobile_data = object_detector(ref_mobile)
mobile_diagonal_in_rf = mobile_data[0][3]  # Diagonal length in pixels

car_data = object_detector(ref_car)
car_diagonal_in_rf = car_data[0][3]  # Diagonal length in pixels

print(f"car diagonal in pixels: {car_diagonal_in_rf}\nMobile diagonal in pixels: {mobile_diagonal_in_rf}")

# Finding focal length
focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_diagonal_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_diagonal_in_rf)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        if d[0] =='car':
            distance = distance_finder(focal_car, CAR_WIDTH, d[1])
            x, y = d[4]
        elif d[0] =='cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[4]
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release() 
