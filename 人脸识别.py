import cv2

#img = cv2.imread('D:/Lena.jpg')

face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(10,10)

while True:
    success, frame = cap.read()
    faces = face_engine.detectMultiScale(frame, 1.3, 5)
    img = frame
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_area = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(face_area,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('frame2', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
