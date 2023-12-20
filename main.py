import cv2
from os import listdir
import os
import numpy as np
from os.path import isfile, join

if __name__ == '__main__':
    data_path='E:/image/python/pythonProject/dataset/'
    onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
    dirs = os.listdir(data_path)
    print(dirs)
    dataset=["","aman","anjali"]
    training_data,lables=[],[]
    for dir_name in dirs:
        subject_dir_path = data_path  + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image in subject_images_names:
            image_path = subject_dir_path +"/"+ image
            print(image_path)
            name = image_path.split('/')[-2]
            print(name.split("_")[0])

            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            training_data.append(np.asarray(images, dtype=np.uint8))
            lables.append(name.split("_")[0])

    print(lables)
    lables = np.asarray(lables,dtype=np.int32)
    model=cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(training_data),np.asarray(lables))
    face_classifier=cv2.CascadeClassifier('E:\image\python\pythonProject\harcascade.xml')
    def face_extractor(img,size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is():
            return img,[]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_face=img[y:y+h,x:x+w]

        return img,cropped_face
    cap=cv2.VideoCapture(0)
    count=0
    while True:
        ret ,frame=cap.read()
        image,face=face_extractor(frame)
        try:
           face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
           result=model.predict(face)
           print(result)

           if result[1]<500:
               confidence=int(100*(1-(result[1])/300))
           if confidence>82:
               cv2.putText(image,dataset[result[0]],(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
               cv2.imshow('facecropped', image)
           else:
               cv2.putText(image, "unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
               cv2.imshow('facecropped', image)
        except:
              cv2.putText(image, "image not", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
              cv2.imshow('facecropped',image)
        if cv2.waitKey(1)==13:
            break

    cap.release()
    cv2.destroyAllWindows()
