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

    # for prediction
    def predictions(self, image):
        row, col, d = image.shape
        # get the YOLO prediction from the image
        # step-1 convert image into square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # step-2: get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        predits = self.yolo.forward()  # detection or prediction from YOLO Model

        # Non Maximum Suppression
        # step-1: filter detection based on confidence (0.4) and probability score (0.25)
        detections = predits[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of the image (photo)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection on object
            if confidence > 0.2:
                class_score = row[5:].max()  # maximum probability from 13 objects
                class_id = row[5:].argmax()  # get the index position at which max probability occur

                if class_score > 0.13:
                    cx, cy, w, h = row[0:4]
                    # construct bounding box from four values
                    # left, top, width and height
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # append values into the list
                    confidences.append(class_score)  # Use class score for the confidence
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.13, 0.45)

        # Check if indices are empty
        if len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []

        # Draw the Bounding Box
        for ind in indices:
            # extract bounding box
            x, y, w, h = boxes_np[ind]
            box_conf = confidences_np[ind]  # Use the confidence directly
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.colors_generate(classes_id)

            text = f'{class_name} {box_conf:.2f}'  # Format the text to show class and confidence
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            
            # Draw the background rectangle for the text
            cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
            
            # Put text with larger size and bold
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            # Print class name
            print(f'Detected: {class_name} with confidence {box_conf:.2f}')

        return image

    # for defining color for prediction box
    def colors_generate(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()

        return tuple(colors[ID])
