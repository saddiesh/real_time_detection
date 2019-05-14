import cv2
import sys
import numpy as np
import os
import tensorflow
import keras.backend
config = tensorflow.ConfigProto()
config.inter_op_parallelism_threads = 1
keras.backend.set_session(tensorflow.Session(config=config))


sys.path.append("/Users/stephaniexia/Documents/lab/Code/Mask_RCNN")
from sample.inference import detect_by_maskrcnn

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

category_index = {}
for class_id in range(len(class_names)):
    category_index[class_id+1] = {class_names[class_id]}

def getTrainingData(window_name, camera_id):
    cv2.namedWindow(window_name) # 创建窗口
    cap = cv2.VideoCapture(camera_id) # 打开摄像头

    color = (0,255,0) #

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        boxes, classes, scores = detect_by_maskrcnn(output_rgb)
        print(boxes, classes, scores)

        boxes = np.squeeze(boxes, axis=0)
        classes = np.squeeze(classes,axis = 0)
        scores = np.squeeze(scores, axis=0)

        print(boxes, classes, len(scores))

        if len(scores) > 0:
            for idx in range(len(scores)):
                x1,y1,x2,y2 = boxes[idx]


                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2) # 画出矩形框
                font = cv2.FONT_HERSHEY_SIMPLEX # 获取内置字体
                cv2.putText(frame, ("{} : {}".format(category_index[classes[idx]], scores[idx])), (x1+30, y1+30), font, 1, (255,0,255), 4) # 调用函数，对人脸坐标位置，添加一个(x+30,y+30）的矩形框用于显示当前捕捉到了多少人脸图片

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap.release()#释放摄像头并销毁所有窗口
    cv2.destroyAllWindows()
    print('Finished.')
#主函数
if __name__ =='__main__':
    print ('catching your face and writting into disk...')
    getTrainingData('getTrainData',0) 