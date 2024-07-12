import cv2
import os
import json
import numpy as np
from data_predit_yolo import YOLO_Pred
from sklearn.metrics import precision_recall_curve, average_precision_score

# Define the paths to your ONNX model and YAML file
onnx_model = 'dataPredictions/Model7/weights/best.onnx'
data_yaml = 'dataPredictions/custom_data.yaml'

# Initialize YOLO_Pred class
yolo = YOLO_Pred(onnx_model, data_yaml)

# Directory containing the test images and their corresponding ground truth labels
test_images_dir = 'test_images'
ground_truth_labels_dir = 'ground_truth_labels'

# Load the ground truth labels
def load_ground_truth_labels(image_id):
    label_file = os.path.join(ground_truth_labels_dir, f'{image_id}.json')
    with open(label_file, 'r') as f:
        labels = json.load(f)
    return labels

# Function to compute mAP
def compute_map(yolo, test_images_dir):
    all_detections = []
    all_ground_truths = []

    # Iterate through test images
    for image_file in os.listdir(test_images_dir):
        image_id, _ = os.path.splitext(image_file)
        image_path = os.path.join(test_images_dir, image_file)
        ground_truth_labels = load_ground_truth_labels(image_id)

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Get YOLO predictions
        detections = yolo.predictions(image)

        # Append ground truth boxes
        ground_truth_boxes = [
            [label['xmin'], label['ymin'], label['xmax'], label['ymax'], label['class_id']]
            for label in ground_truth_labels
        ]
        all_ground_truths.append(ground_truth_boxes)

        # Append detected boxes
        detected_boxes = []
        for detection in detections:
            x, y, w, h, class_id, confidence = detection
            xmin = int(x - w / 2)
            ymin = int(y - h / 2)
            xmax = int(x + w / 2)
            ymax = int(y + h / 2)
            detected_boxes.append([xmin, ymin, xmax, ymax, class_id, confidence])
        all_detections.append(detected_boxes)

    # Compute precision-recall and average precision for each class
    average_precisions = {}
    for class_id in range(len(yolo.labels)):
        true_positives = []
        scores = []
        num_gt = 0

        for gt_boxes, det_boxes in zip(all_ground_truths, all_detections):
            gt_boxes_class = [box for box in gt_boxes if box[4] == class_id]
            det_boxes_class = [box for box in det_boxes if box[4] == class_id]
            num_gt += len(gt_boxes_class)

            # Sort detections by confidence
            det_boxes_class.sort(key=lambda x: x[5], reverse=True)

            assigned_gt = []
            for det in det_boxes_class:
                scores.append(det[5])
                if not gt_boxes_class:
                    true_positives.append(0)
                    continue
                ious = [iou(det, gt) for gt in gt_boxes_class]
                max_iou = max(ious)
                max_index = ious.index(max_iou)
                if max_iou >= 0.5 and max_index not in assigned_gt:
                    true_positives.append(1)
                    assigned_gt.append(max_index)
                else:
                    true_positives.append(0)

        if num_gt == 0:
            continue

        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum([1 - x for x in true_positives])

        recall = true_positives / num_gt
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        average_precisions[class_id] = average_precision_score(recall, precision)

    mAP = np.mean(list(average_precisions.values()))
    return mAP, average_precisions

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Run mAP calculation
mAP, class_wise_ap = compute_map(yolo, test_images_dir)
print(f'mAP: {mAP:.4f}')
for class_id, ap in class_wise_ap.items():
    print(f'Class {class_id} ({yolo.labels[class_id]}): AP = {ap:.4f}')
