## 📌 Project Overview
This project sets up real-time object detection using YOLO (You Only Look Once) and a web-based dashboard to visualize detected objects. The system:
- Uses YOLOv8 (or YOLOv10) for live detection via webcam.
- Stores detected objects in a local NoSQL database (TinyDB).
- Provides real-time bar and pie charts using D3.js to track object counts.

## 📌 Installation & Setup
1. Install Dependencies

    Make sure you have Python 3.8+ installed. Then, install the required packages:

    ```
    pip install ultralytics opencv-python numpy tinydb
    ```
2. Run the YOLO Webcam Script

    This script detects objects using your webcam and stores results in detections.json.

    ```
    python yolo.py
    ```
3. Start the Web Dashboard

    Run a local server to view the dashboard:

    ```
    python -m http.server 8000
    ```

Then, open:
http://localhost:8000/index.html
