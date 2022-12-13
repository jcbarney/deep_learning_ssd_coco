import cv2
# import tensorflow as tf
import numpy as np
# import PIL
import matplotlib.pyplot as plt
import os
import datetime

#set up model
#config and frozen graph copied from https://github.com/ankityddv/ObjectDetector-OpenCV
config = "model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model_path = "model/frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(model_path,config)
model.setInputSize(512, 512)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputScale(1.0/127.5)

# get class label mapping
import json
file_name = 'data/annotations/label_map.json'
f = open(file_name)
labels = json.load(f)
f.close()

# test image
# test = np.asarray(PIL.Image.open('images/dog.jpeg'))
# class_pred, confidence, boxes = model.detect(test, confThreshold=0.5)
# font = cv2.FONT_HERSHEY_DUPLEX
# for class_ix, conf, box in zip(class_pred, confidence, boxes):
#     cv2.rectangle(test, box, (255, 0, 0), 2)
#     cv2.putText(test, labels[str(class_ix)], (box[0]+10, box[1]+40), font, fontScale=5, color=(0, 255, 0), thickness=5)

# test time to predict 1000 images
files=os.listdir('data/test2017')

import random
random_ix = random.sample(range(len(files)), 1000)

images = []
for i in random_ix:
    current = np.asarray(PIL.Image.open(f'data/test2017/{files[i]}'))
    images.append(current)

start = datetime.datetime.now()
for im in images:
    model.detect(im, confThreshold=0.5)
end = datetime.datetime.now()

#results
print(end-start)