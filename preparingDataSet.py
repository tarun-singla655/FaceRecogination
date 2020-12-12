
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
        
import csv        
face_id=int(input('enter your id '))
FirstName = input("Enter you firstname ")
LastName = input("Enter you firstname ")
camera_port = 0
vid_cam = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW);


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


count = 0

assure_path_exists("dataset/")

# Start looping
while(True):
    _,image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count>=100:
        print("Successfully Captured")
        with open('record.csv','a') as apendobj:
            append = csv.writer(apendobj)
            append.writerow([id,FirstName,LastName])
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()