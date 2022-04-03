import tensorflow as tf
from cvzone.FaceMeshModule import FaceMeshDetector
import tensorflow_hub as hub
import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
model=hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet=model.signatures['serving_default']
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold )
        draw_keypoints(frame, person, confidence_threshold)
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4,(0,255,0), -1)
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1,  c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1> confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)) , ( int(x2), int(y2)), (0,0,255), 2)  
EDGES = {
    (0,1): 'm',
    (0,2):'c',
    (1,3):'m',
    (2, 4):'c',
    (0, 5):'m',
    (0, 6):'c',
    (5, 7):'m',
    (7, 9):'m',
    (6, 8):'c',
    (8, 10):'c',
    (5, 6):'y',
    (5, 11):'m',
    (6, 12):'c',
    (11, 12):'y',
    (11, 13):'m',
    (13, 15):'m',
    (12, 14):'c',
    (14, 16):'c'
}
# model= Sequential()
detector = FaceMeshDetector(maxFaces=2)

def predict_classes(self, x, batch_size=32, verbose=1):
        
        
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-3] > 1:
            return proba.argmax(axis=-3)
        else:
            return (proba > 0.5).astype('int32') 



facedetect = cv2.CascadeClassifier("C:/Users/HP/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_SIMPLEX
# model = load_model("D:/facial_recognition/images/anand_chutiya.h5")
model = load_model("D:/facial_recognition/images/new_model.h5")
classes = ['ANAND', 'PUJA','PRITI']
print(classes[0])

def get_className(classNo):
    if classNo == 0:
        return"anand"
    elif classNo==1:
        return"puja"
    elif classNo==2:
        return"priti"
    elif classNo==3:
        return"NONE"
while True:
    sucess, imgOriginal= cap.read()
    imgOriginal, faces = detector.findFaceMesh(imgOriginal)
    if faces:
        print(faces[0])
        
    
    faces = facedetect.detectMultiScale(imgOriginal, 1.3,5)
    for x,y,w,h in faces:
        crop_img=imgOriginal[y:y+h,x:x+w]
        img=cv2.resize(crop_img, (224,224))
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192,256)
        input_img = tf.cast(img, dtype= tf.int32)
        img=img.reshape(1, 224, 224,3)
        results = movenet(input_img)
        
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].   reshape((6,17,3)) 
        loop_through_people(imgOriginal, keypoints_with_scores, EDGES, 0.3)
        print(keypoints_with_scores)
        prediction = model.predict(img)
        # model=sequential()
        # predictions = model.predict((img.shape[0],img.shape[1],img.shape[2]))

        # classIndex= classes[np.argmax(img)]
        classIndex = model.predict_classes(img)
        # classIndex=model.classIndex1
        # np.argmax(model.predict(x), axis=-1)
        # classIndex = np.argmax(model.predicted_classes(img), axis=-1)
        probabilityValue = np.amax(prediction)
        if classIndex==0:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        elif classIndex==1:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        elif classIndex==2:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        elif classIndex==3:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        cv2.putText(imgOriginal,str(round(probabilityValue*100, 2))+'%', (180, 95), font, 0.75, (255,255,0))
        
    cv2.imshow("RESULT",imgOriginal)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()