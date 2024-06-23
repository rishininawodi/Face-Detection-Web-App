import cv2

#Read vedio stream from webcam 
#So create object
video_capture = cv2.VideoCapture(0)  #Put zero if using default laptop or desktop web cam..Other wise use  exteranal cam put -1 or 1

#create infinite loop
while True:
    _, img = video_capture.read()  #read vedio as an image.normal returns two parameters.we only use image so used " _, "
     #check web cam is working
    cv2.imshow("face detection" , img)  #face detection mean name for the window
    if cv2.waitKey(1) & 0xFF == ord('e'):#break the loop if user press q.So terminating condition for loop
        break

video_capture.release()  #release web cam 
cv2.destroyAllWindows() #cv2 destroyed from all the wibdows

