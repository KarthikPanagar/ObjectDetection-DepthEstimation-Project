import cv2 as cv
import numpy as np

# YOLOv4 configuration
yolov4_weights = 'yolov4-tiny.weights'
yolov4_config = 'yolov4-tiny.cfg'
yolov4_classes = 'classes.txt'

# Load YOLOv4 model
net = cv.dnn.readNet(yolov4_weights, yolov4_config)

# Load COCO class labels
classes = []
with open(yolov4_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Load reference image
ref_image = cv.imread('15c.jpg')

# Preprocess the reference image for YOLOv4
blob = cv.dnn.blobFromImage(ref_image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Get output layer names
out_layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass to get object detections
outs = net.forward(out_layer_names)

# Process output to get bounding boxes, confidences, and class IDs
conf_threshold = 0.5
nms_threshold = 0.4

height, width = ref_image.shape[:2]

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold and classes[class_id] == 'car':
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Perform non-maximum suppression to eliminate redundant overlapping boxes
indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw rectangle box around the car if detected
if len(indices) > 0:
    box = boxes[indices[0]]
    x, y, w, h = box
    cv.rectangle(ref_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the reference image with rectangle box around the car
    cv.namedWindow('Reference Image', cv.WINDOW_NORMAL)
    cv.imshow('Reference Image', ref_image)
    cv.waitKey(0)

    # Calculate diagonal distance in pixels
    diagonal_distance_pixels = np.sqrt(w**2 + h**2)
    print(f"Diagonal distance in pixels: {diagonal_distance_pixels:.2f}")
else:
    print("person not detected in the reference image.")
