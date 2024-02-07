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

# Function to perform object detection and calculate diagonal distance
def detect_and_calculate_distance(frame):
    # Preprocess the frame for YOLOv4
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Get output layer names
    out_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get object detections
    outs = net.forward(out_layer_names)

    # Process output to get bounding boxes, confidences, and class IDs
    conf_threshold = 0.5
    nms_threshold = 0.4

    height, width = frame.shape[:2]

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

    # Draw rectangle box around the person if detected and calculate diagonal distance in pixels
    if len(indices) > 0:
        box = boxes[indices[0]]
        x, y, w, h = box
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate diagonal distance in pixels
        diagonal_distance_pixels = np.sqrt(w**2 + h**2)
        print(f"Diagonal distance in pixels: {diagonal_distance_pixels:.2f}")

        # Calculate estimated distance (Y) using the equation Y = -0.05203*X + 18.67
        estimated_distance = -0.05203 * diagonal_distance_pixels + 18.67


        # Display the estimated distance on top of the bounding box
        cv.putText(frame, f"Estimated Distance: {estimated_distance:.2f}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the frame with the rectangle box and estimated distance
    cv.imshow('Live Feed', frame)

# Start capturing from the live camera feed
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_calculate_distance(frame)

    # Press 'q' to exit
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
