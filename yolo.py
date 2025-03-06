import cv2
import time
from ultralytics import YOLO
from tinydb import TinyDB, Query

# Initialize TinyDB (JSON-based NoSQL)
db = TinyDB("detections.json")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    # Extract detected objects
    detections = []
    for obj in results[0].boxes.data.tolist():  
        x1, y1, x2, y2, conf, class_id = obj
        class_name = model.names[int(class_id)]
        detections.append(class_name)

    # Store detections in TinyDB
    if detections:
        db.insert({"timestamp": time.time(), "objects": detections})

    # Display detections on frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Webcam", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
