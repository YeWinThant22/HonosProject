import cv2
from data_predit_yolo import YOLO_Pred

# set define after training model and yaml file 
onnx_module = 'dataPredictions/Model7/weights/best.onnx'
data_yaml = 'dataPredictions/custom_data.yaml'

yolo = YOLO_Pred(onnx_module,data_yaml)

# define video file
cap = cv2.VideoCapture('dataPredictions/video.mp4')

while True:
    ret, frame = cap.read()
    if ret == False:
        print('unable to read video')
        break

    pred_image = yolo.predictions(frame)

    cv2.imshow('YOLO',pred_image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()