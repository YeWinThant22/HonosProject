import cv2
import time
from data_predit_yolo import YOLO_Pred

# Define the paths to your ONNX model and YAML file
onnx_model = 'dataPredictions/Model7/weights/best.onnx'
data_yaml = 'dataPredictions/custom_data.yaml'

# Initialize YOLO_Pred class
yolo = YOLO_Pred(onnx_model, data_yaml)

# Start video capture from external webcam (1 for external)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the window to full screen
cv2.namedWindow('Webcam Prediction', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Webcam Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predictions
    img_pred = yolo.predictions(frame)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Check if any predictions were made
    if len(img_pred) > 0:  # If there are predictions
        # Display FPS on frame
        cv2.putText(img_pred, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam Prediction', img_pred)
    else:
        # Show an empty frame or a message indicating no detections
        cv2.putText(frame, 'No objects detected', (frame.shape[1] // 2 - 100, frame.shape[0] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
