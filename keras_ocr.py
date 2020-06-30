import cv2 
import keras_ocr 

image = cv2.imread('image/address')

pipeline = keras_ocr.pipeline.Pipeline()

prediction = pipeline.recognize([image])

for i in prediction[0]:
    print(i[0], end=' ') 
