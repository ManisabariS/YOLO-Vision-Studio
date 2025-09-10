import cv2
import os
import numpy as np

# ==============================
# PATCH for headless OpenCV (no GUI)
if not hasattr(cv2, "imshow"):
    def imshow(*args, **kwargs):
        # just a placeholder so ultralytics doesn't crash
        pass
    cv2.imshow = imshow
# ==============================

from ultralytics import YOLO

# ==============================
# CONFIG VARIABLES
MODEL_PATH = "raghu_model.pt"  # Path to your YOLO model file
OUTPUT_FOLDER = "annotated_frames"  # Folder to save annotated frames
# ==============================

# Load model
model = YOLO(MODEL_PATH)

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 = default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 90)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = model(frame)
    annotated_frame = results[0].plot()

    filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count}.jpg")
    cv2.imwrite(filename, annotated_frame)
    frame_count += 1
    cv2.imshow("Annotated Frame", annotated_frame)
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()