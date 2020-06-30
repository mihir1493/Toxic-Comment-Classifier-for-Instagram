import cv2 
import keras_ocr 

image = cv2.imread('/home/devesh/Downloads/flag.jpeg')

pipeline = keras_ocr.pipeline.Pipeline()

prediction = pipeline.recognize([image])

for i in prediction[0]:
    print(i[0], end=' ') 
