import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

# Load YAML file
with open('dataPredictions/custom_data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml['names']

# Load YOLO Model
yolo = cv2.dnn.readNetFromONNX('dataPredictions/Model7/weights/best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the image
img = cv2.imread('dataPredictions/photo.jpg')
if img is None:
    print("Error: Image not found or could not be opened.")
    exit()

image = img.copy()
row, col, d = image.shape

# Get the YOLO prediction from the image
max_rc = max(row, col)
input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
input_image[0:row, 0:col] = image

INPUT_WH_YOLO = 640
blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
yolo.setInput(blob)
predits = yolo.forward()

# Non-Maximum Suppression
detections = predits[0]
boxes = []
confidences = []
classes = []

image_w, image_h = input_image.shape[:2]
x_factor = image_w / INPUT_WH_YOLO
y_factor = image_h / INPUT_WH_YOLO

for i in range(len(detections)):
    row = detections[i]
    confidence = row[4]
    if confidence > 0.2:
        class_score = row[5:].max()
        class_id = row[5:].argmax()

        if class_score > 0.25:
            cx, cy, w, h = row[0:4]
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            box = np.array([left, top, width, height])
            confidences.append(confidence)
            boxes.append(box)
            classes.append(class_id)

boxes_np = np.array(boxes).tolist()
confidences_np = np.array(confidences).tolist()

index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

# Draw the Bounding Box
for ind in index:
    x, y, w, h = boxes_np[ind]
    boxb_conf = int(confidences_np[ind] * 100)
    classes_id = classes[ind]
    class_name = labels[classes_id]

    text = f'{class_name}: {boxb_conf}%'
    
    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw the background rectangle for the text
    cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
    
    # Put text with larger size and bold
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)


cv2.imshow('original', img)
cv2.imshow('data_predictions_yolo', image)

# Wait until a key is pressed and then close the windows
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
