import cv2, numpy as np
import pandas as pd
names = pd.read_csv("record.csv")
print(names)
EmpId = []
Names = []
for i in names.index:
    EmpId.append(names['Employ_id'][i])
    Names.append([names['FirstName'][i] , names['LastName'][i] ] )
# import xlwrite
import time
import sys
start=time.time()
period=8
camera_port = 0
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainer/trainer.yml');
flag = 0;
id=0;
filename='filename';
dict = {}
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.2, 5);
  
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        id,result=recognizer.predict(roi_gray)
        if(result < 500):
            print(id)
            Firstname = Names[EmpId.index(id)][0]
            Lastname  = Names[EmpId.index(id)][1]
              
            conf = int(100*(1-(result)/300))
            if conf > 75:
                ID = Firstname + " " + Lastname
            else:
                ID = 'Unknown, can not recognize'
        else:
             print("not found")
             ID = 'Unknown, can not recognize'
             flag=flag+1
             break
        cv2.putText(img,str(ID)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
        # cv2.putText(img,str(id),(x,y+h),font,0.55,(0,0,255),1);
    cv2.imshow('frame',img);
    if time.time()>start+period:
        break;
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cap.release();
cv2.destroyAllWindows();