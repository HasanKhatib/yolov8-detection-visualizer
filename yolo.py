import cv2
import time
import argparse
from ultralytics import YOLO
from tinydb import TinyDB

# Initialize TinyDB (JSON-based NoSQL)
db = TinyDB("detections.json")

model = YOLO("yolov8n.pt")

# selct mode and image
parser = argparse.ArgumentParser(description="Run YOLO on camera or image.")
parser.add_argument("--mode", choices=["cam", "img"], required=True, help="Select input source: 'cam' for webcam, 'img' for an image file.")
parser.add_argument("--image", type=str, help="Path to image file (required for 'img' mode).")
args = parser.parse_args()

def process_frame(frame):
    """Runs YOLO on a frame/image and saves results."""
    results = model(frame)

    detections = []
    for obj in results[0].boxes.data.tolist():  
        x1, y1, x2, y2, conf, class_id = obj
        class_name = model.names[int(class_id)]
        detections.append(class_name)

    # persist to db
    if detections:
        db.insert({"timestamp": time.time(), "objects": detections})

    return results[0].plot()  # Returns annotated image/frame

if args.mode == "cam":
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
        # process and show frame
        annotated_frame = process_frame(frame)
        cv2.imshow("YOLO Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

elif args.mode == "img":
    if not args.image:
        print("Error: You must provide an image path with --image when using 'img' mode.")
        exit()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not load image '{args.image}'.")
        exit()

    # process and show image
    annotated_frame = process_frame(frame)
    cv2.imshow("YOLO Image Detection", annotated_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
