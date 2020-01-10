# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:05:37 2019

@author: user
"""

import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_frontalface_default.xml')


# extract the face features

def face_extractor(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces=face_classifier.detectMultiScale(gray,1.3,5)        #here 1.3 is scaling factor,and  5 is neighour ye value 3-6 ke bich honi chahiye for accuray age low rakhenge (3) to accuracy thik nhi ayegi...
    
    
    if faces is():
        return None
    
    for(x,y,w,h) in faces:
        cropped_face=img[y:y+h,x:x+w]
        
    return cropped_face


       
cap=cv2.VideoCapture(0)
count=0    # this variable for counting the no of faces app kitne photo lena chahte h


while True:
    ret,frame=cap.read()

    if face_extractor(frame) is  None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))    #yaha pe face ko resize kr rhe h qki hm chahte hai ki face ka size uta hi ho jitni camera ki window ka size.......
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)         #image ko gray me change kr rhe h
        
        # yaha pe uska path denge jaha pe photo click hoke hm save krenge..
        file_name_path='E:\python practice\openCV\collected face/user' +str(count)+'.jpg'
        cv2.imwrite( file_name_path,face)           #yaha pe photo save ho rhi h
        
        #
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)     #last parameter font size h 2
     
        cv2.imshow('face Cropper',face)
        
    else:
        print('Face not found')
        pass
    
        
     
    if cv2.waitKey(1)==13 or count==100:
        break
    
    
    
    
cap.release()
cv2.destroyAllWindows()
print('Collecting Sampling Complete!!!')
        
        
        
        












