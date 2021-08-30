import cv2
import numpy as np
import os

#Given an image below function returns rectangle for face detected alongwith gray scale image.
def faceDetection(test_img):
    gray_image= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces= face_cascade.detectMultiScale(image=gray_image, scaleFactor=1.3, minNeighbors=5)

    return faces, gray_image

#Given a directory below function returns part of gray_image which is face alongwith its label/ID
def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print('Skipping system file') #Skipping files that startwith .
                continue

            id= os.path.basename(path) #fetching subdirectory names
            img_path= os.path.join(path, filename) #fetching image path
            print('img_path:', img_path)
            print('id:', id)
            test_img= cv2.imread(img_path) #loading each image one by one
            if test_img is None:
                print('Image not loaded properly')
                continue
            faces_rect, gray_image=faceDetection(test_img) #Calling faceDetection function to return faces detected in particular image
            #If there are two faces in a single image then we will skip that image.
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)= faces_rect[0]
            RoI_gray= gray_image[y:y+w, x:x+h] #cropping region of interest i.e. face area from grayscale image
            faces.append(RoI_gray)
            faceID.append(int(id))
    return faces, faceID

#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments.
def train_classifier(faces, faceID):
    face_recognizer= cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

#Below function draws bounding boxes around detected face in image.
def draw_rect(test_img, face):
    (x,y,w,h)= face
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), 5)

#Below function writes name of person for detected label.
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_COMPLEX, 5, (0,255,0), 4, cv2.LINE_AA)