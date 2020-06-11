import cv2, time
first_frame = None
video = cv2.VideoCapture('./Video.mp4')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    check, img = video.read()
    img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05, minNeighbors = 18)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Jay",img)
    key = cv2.waitKey(1)
    if key==ord("q"):
        break

video.release()
cv2.destroyAllWindows()