import tensorflow.keras
import numpy as np
import cv2


class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        np.set_printoptions(suppress=True)
        self.model = tensorflow.keras.models.load_model(self.model_path)

        
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = []
            for line in label_file:
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw= True, pos=(50, 50), scale=2, color = (0,255,0)):
        
        imgS = cv2.resize(img, (224, 224))
        
        image_array = np.asarray(imgS)
       
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

       
        self.data[0] = normalized_image_array

        
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal



def main():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier("D:/recognitionimage/converted_keras (3)/keras_model.h5", "D:/recognitionimage/converted_keras (3)/labels.txt")
    while True:
        _, img = cap.read()
        predection = maskClassifier.getPrediction(img)
        print(predection)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
