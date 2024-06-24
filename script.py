import cv2

#define a method to draw the boundary around the features
def draw_boundary(img,classifire,scaleFactor,minNeighbors,color,text):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Our RGB image  convert to gray_scale image
    features = classifire.detectMultiScale(gray_img,scaleFactor,minNeighbors)   #finding/detecting  features in classifires

    #declare the list which will hold the cordinates for face.
    coords = []
    for( x,y,w,h) in features:
        cv2.rectangle(img ,(x,y) ,(x+w , y+h) , color , 2)
        cv2.putText(img,text,(x, y-4), cv2.FONT_HERSHEY_COMPLEX, 0.8,color,1,cv2.LINE_AA) #text type cordinates
        coords = [x , y, w,h]  #update the cordinatees

    return coords,img    #return the cordinates and updated it to image   

def detect(img,caceCascade): #pass face and image
    color = { "blue":(255,0,0), "red":(0,0,255), "green":(0,255,0) }

    coords,img = draw_boundary(img, faceCascade , 1.1 , 10 , color['blue'] ,"face") #call function
    return img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



#Read vedio stream from webcam 
#So create object
video_capture = cv2.VideoCapture(0)  #Put zero if using default laptop or desktop web cam..Other wise use  exteranal cam put -1 or 1

#create infinite loop
while True:
    _, img = video_capture.read()  #read vedio as an image.normal returns two parameters.we only use image so used " _, "
    img = detect(img, faceCascade)
     #check web cam is working
    cv2.imshow("face detection" , img)  #face detection mean name for the window
    if cv2.waitKey(1) & 0xFF == ord('e'):#break the loop if user press q.So terminating condition for loop
        break

video_capture.release()  #release web cam 
cv2.destroyAllWindows() #cv2 destroyed from all the wibdows

