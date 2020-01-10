# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:16:25 2019

@author: user
"""

import cv2
import numpy as np
from os import listdir             #here this lib we r using becoz we have to fetch data from another file...
from os.path import isfile, join




data_path='E:\python practice\openCV\collected face/'         #this is path where our image is store we becoz w e need images for traing
onlyfiles=[f for f in listdir(data_path,) if isfile(join(data_path,f))]   #we also need files and we join path and files


Training_Data, Labels=[],[]          #data hmare pas list ki form me ayega isliye list use kr rhe h

for i,files in enumerate(onlyfiles):
    image_path= data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))   #now jo image arhi h unko append krenge apne list me..
    Labels.append(i)
    
    
Labels=np.asarray(Labels,dtype=np.int32)   #here hm array ko cal kr rhe h uske ander Labels ur uska type h

#build the model

model=cv2.face.LBPHFaceRecognizer_create()   #linear binary face histogram face recognizer...

#now hum model ko traing krenge jo arry ki form me hai...
model.train(np.asarray(Training_Data),np.asarray(Labels))

print("Model Traing Complete!!!!")





face_classifier=cv2.CascadeClassifier('https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_frontalface_default.xml')




# now we have to detect face....qki hme check krna hoga ki ye wahi face hai ki nahi...


def face_detector(img,size=0.5):
    #jo bhi img milegei camera ke frame se use gray scale convert kerenge..
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is():            #here agr face nhi milta to simple imgage and khali list return krenge
        return img,[]
    #here we find face age face mil jata to kya krenge..ek rectanle bnayenge and uska roi(region of interest calculate krenge qki hamara interese face me hai) x,y coordinate hai ur width height me kitna change ayega usek liye
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))             #jo image mili h usko resize karna to bnata..
    
    return img,roi





# now main logic
    
cap=cv2.VideoCapture(0)
while True:
    
    ret,frame=cap.read()
    
    image,face= face_detector(frame)  # jo image mil rhi hai wo face detector se mil rhi h
    
    try:
        
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)        #jo face ki image mil rhi h usko gray me convert krenge...
        result=model.predict(face)    # now ab model(face) ko predict krenge ki ye wahi face hai ki nahi...
        
        if result[1]<500:         #here it is just a psudeo value app koi bhi le skte..
            confidence=int(100(1-(result[1])/300))      #here we are calculating confidance value.. 1 se isliye subtract kr rhe h qki python value me 1 ka diff rahta ur 100 se multiply exact percentage ke liye kr rhe h
            display_string=str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,129,255),2)  # hme text image ke uper chahiye..ur kaha yani kis coordinate me chahiye,then font and font color and font ki size
            
            
            
            
            if confidence>75:
                cv2.putText(image,"Unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  # age value 75 se jyada means face match kr rha h hme text image ke uper chahiye..ur kaha yani kis coordinate me chahiye,then font and font color and font ki size
                cv2.imshow('Face Cropper',image)   #facemilne pr cropped face bhi ana chahiye so sho krwa rhe h
                
            else:
                cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)  # age value 75 se km h means face match nahi kr rha h hme text image ke uper chahiye..ur kaha yani kis coordinate me chahiye,then font and font color and font ki size
                cv2.imshow('Face Cropper',image)   #facemilne pr cropped face bhi ana chahiye so sho krwa rhe h
                
                
    except:
        cv2.putText(image,"Face not found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2) 
        cv2.imshow('Face Croppe',image)
        pass
 
    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()





























