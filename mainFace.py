import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from GenderClassifier import GenderClassifier

def analyzeFace(model,img_face):
    img_face = np.array(img_face)
    imageSize = img_face.shape
    img_face = cv2.resize(img_face, (128, 128))
    img_face = cv2.cvtColor(img_face,cv2.COLOR_BGR2GRAY)
    img_face = img_face.reshape(128,128,1)

    # 1 = Male; 0 = Female
    predict_gender = model.predict([img_face])
    gender = np.array(predict_gender)
    gender = predict_gender.argmax()
    if gender==0:
        gender = 'Female'
    else:
        gender = 'Male'

    return imageSize,gender

def load_cascade():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascadeFiles/haarcascade_eye.xml')
    return face_cascade,eye_cascade

def webcamToggle(switch):
    video_path = 'C:/Users/Nahid/Documents/#Movies/The Good,the Bad and the Ugly 1966 720p (Bia2Movies).mkv'
    if switch == True:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(video_path)
    return cam


def main():
    ob_classifier = GenderClassifier(128,128)
    model = ob_classifier.get_model()

    face_cascade,eye_cascade = load_cascade()
    cam = webcamToggle(True)


    frameNumber = 0
    while True:
        ret,img = cam.read()
        frameNumber = frameNumber + 1

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle( img, (x,y), (x+w,y+h), (255,255,0), 1 )
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            ret_size,gender = analyzeFace(model,roi_color)
            cv2.putText(img,gender,(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0))
            '''
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0) ,1)
            '''

        cv2.imshow('WebCam',img)
        #cv2.imshow('faceImage',roi_color)
        print('Frame Number = ' + str(frameNumber) + ' -> Face_shape -> ' + str(ret_size))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__== "__main__":
  main()
