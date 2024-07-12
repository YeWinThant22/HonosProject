#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Load YAML file
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO Model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        predits = self.yolo.forward()

        detections = predits[0]
        boxes = []
        confidences = []
        classes = []

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection on object
            if confidence > 0.35:  # Adjust confidence threshold
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > 0.35:  # Adjust class score threshold
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])
                    confidences.append(class_score)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Non-Maximum Suppression
        nms_threshold = 0.55
        indices = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.35, nms_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []

        for ind in indices:
            x, y, w, h = boxes_np[ind]
            box_conf = confidences_np[ind]
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.colors_generate(classes_id)

            text = f'{class_name} {box_conf:.2f}'
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            print(f'Detected: {class_name} with confidence {box_conf:.2f}')

        return image

    def colors_generate(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])

# Main execution for webcam
if __name__ == "__main__":
    onnx_model_path = "path_to_your_model/best.onnx"  # Update with your model path
    data_yaml_path = "path_to_your_yaml/data.yaml"   # Update with your YAML path

    yolo_pred = YOLO_Pred(onnx_model_path, data_yaml_path)

    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = yolo_pred.predictions(frame)
        cv2.imshow('YOLO Object Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
