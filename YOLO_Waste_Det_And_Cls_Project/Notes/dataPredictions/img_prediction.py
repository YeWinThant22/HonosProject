import cv2
from data_predit_yolo import YOLO_Pred

# set define after training model and yaml file 
onnx_module = 'dataPredictions/Model7/weights/best.onnx'
data_yaml = 'dataPredictions/custom_data.yaml'

yolo = YOLO_Pred(onnx_module,data_yaml)


img = cv2.imread('dataPredictions/waste.jpg')

# Predictions

img_pred = yolo.predictions(img)

if img_pred is not None:
    cv2.imshow('prediction image',img_pred)
    # Wait until a key is pressed and then close the windows
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cv2.destroyAllWindows()
    print("Prediction Successful.")
else:
    print("Error")