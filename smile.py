# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:54:10 2021

@author: Aizaz
"""

import cv2
import random

face_train =cv2.CascadeClassifier("D:/install/artificial intelligence_install/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
smile_train =cv2.CascadeClassifier("D:/install/artificial intelligence_install/Anaconda3/Lib/site-packages/cv2/data/haarcascade_smile.xml")
cam = cv2.VideoCapture(0)
while True:
    (check ,frame) = cam.read()
    bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    face = face_train.detectMultiScale(bw)
  
    
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y),(w+x,h+y),(random.randrange(256),random.randrange(256),random.randrange(256)),2)
        
        the_face=frame[y:y+h,x:x+h]
        bw = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
        smile =smile_train.detectMultiScale(bw,scaleFactor =1.7,minNeighbors=20)
        
        #for (xz,yz,wz,hz) in smile:
          #  cv2.rectangle(the_face, (xz,yz),(wz+xz,hz+yz),(random.randrange(256),random.randrange(256),random.randrange(256)),2)
    if len(smile)>0:
        cv2.putText(frame,"smilimg",(x,y+h+40),fontScale = 2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,255))
    cv2.imshow("pic",frame)
    k=cv2.waitKey(1)
    if k==65 or k==97:
        break
cam.release()
cv2.destroyAllWindows()