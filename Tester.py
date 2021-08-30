import cv2
import numpy as np
import os
import faceRecognition as fr

test_img= cv2.imread('C:/Users/maya/Desktop/OpenCV Python/testImages/img8.jpg')
faces_detected, gray_img= fr.faceDetection(test_img)
print('faces_detected:', faces_detected)

#for (x,y,w,h) in faces_detected:
#    cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,0,255), 3)

faces,faceID= fr.labels_for_training_data('C:/Users/maya/Desktop/OpenCV Python/trainingImages')
face_recognizer= fr.train_classifier(faces, faceID)
face_recognizer.write('trainingData.yml') 
name= {0:'Hardik', 1:'Fenil'}

for face in faces_detected:
    (x,y,w,h)= face
    RoI_gray= gray_img[y:y+h, x:x+w]
    label,confidence= face_recognizer.predict(RoI_gray)
    print('confidence:',confidence)
    print('label:',label)
    fr.draw_rect(test_img, face)
    predicted_name= name[label]
    if (confidence>37):
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(500,500))
cv2.imshow('face detection', resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()