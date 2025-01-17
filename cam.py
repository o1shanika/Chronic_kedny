import cv2
import time

# Load YOLO model and its configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO dataset object names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Timer setup
interval = 300  # 5 minutes in seconds
start_time = time.time()
pedestrian_count = 0  # To store the number of pedestrians detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # YOLO expects a 416x416 image, so we resize and create a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the YOLO output
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through each output layer
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()  # Get the index of the highest score (the predicted class)
            confidence = scores[class_id]

            # Only consider detections of "person" with a confidence threshold
            if class_id == 0 and confidence > 0.5:  # "person" has class_id = 0 in COCO dataset
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Get top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append bounding box, confidence, and class ID
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to avoid overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Loop through remaining boxes and draw rectangles for detected people
    pedestrian_count_in_frame = 0
    if len(indexes) > 0:
        for i in indexes.flatten():
            if class_ids[i] == 0:  # Person class
                x, y, w, h = boxes[i]
                pedestrian_count_in_frame += 1
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the pedestrian count on the video feed
    cv2.putText(frame, f"Pedestrians: {pedestrian_count_in_frame}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with detections
    cv2.imshow("Pedestrian Detection", frame)

    # Count the pedestrians every 5 minutes
    if time.time() - start_time >= interval:
        pedestrian_count += pedestrian_count_in_frame
        print(f"Pedestrian count in the last 5 minutes: {pedestrian_count}")

        # Reset the timer and pedestrian count for the next interval
        start_time = time.time()
        pedestrian_count = 0

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
