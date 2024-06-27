import cv2

#this part for generating dataset to train classifire...
def generate_dataset(img,id,img_id):
    cv2.imwrite("data/user." + str(id) + "."+str(img_id)+".jpg",img)

#define a method to draw the boundary around the features
def draw_boundary(img,classifire,scaleFactor,minNeighbors,color,text,clf):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Our RGB image  convert to gray_scale image
    features = classifire.detectMultiScale(gray_img,scaleFactor,minNeighbors)   #finding/detecting  features in classifires

    #declare the list which will hold the cordinates for face.
    coords = []
    for( x,y,w,h) in features:
        cv2.rectangle(img ,(x,y) ,(x+w , y+h) , color , 2)
        id, _ = clf.predict(gray_img[y:y+h , x:x+w])

        if id ==1:    

            cv2.putText(img,"Ali",(x, y-4), cv2.FONT_HERSHEY_COMPLEX, 0.8,color,1,cv2.LINE_AA) #text type cordinates
        coords = [x , y, w,h]  #update the cordinatees

    return coords   #return the cordinates and updated it to image   

def recognize(img,clf,faceCascade):
    color = { "blue":(16, 165, 173 ), "red":(0,0,255), "green":(0,255,0) , "pink":(244, 37, 162 ) }

    coords = draw_boundary(img,faceCascade,1.1,10,color["blue"], "Face" , clf)

def detect(img,faceCascade,eyeCascade,noseCascade,mouthCascade,img_id): #pass face and image
    color = { "blue":(16, 165, 173 ), "red":(0,0,255), "green":(0,255,0) , "pink":(244, 37, 162 ) }

    coords = draw_boundary(img, faceCascade , 1.1 , 10 , color['blue'] ,"face") #call function

    return img
    #Check  if the length of this equal to 4.since we returm four cordinates above..
    if len(coords) ==4:

        roi_img = img[coords[1] : coords[1]+coords[3] , coords[0]:coords[0]+coords[2]]
        user_id =1
        generate_dataset(roi_img,user_id,img_id)

        
        #coords = draw_boundary(roi_img, eyeCascade , 1.1 , 14 , color['pink'] ,"eye")
        #coords = draw_boundary(roi_img, noseCascade , 1.1 , 5 , color['green'] ,"nose")
        #coords = draw_boundary(roi_img, mouthCascade , 1.1 , 20 , color['red'] ,"mouth") 


    return img

#detecting Face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifire.yml")


#Read vedio stream from webcam 
#So create object
video_capture = cv2.VideoCapture(0)  #Put zero if using default laptop or desktop web cam..Other wise use  exteranal cam put -1 or 1

img_id =0

#create infinite loop
while True:
    _, img = video_capture.read()  #read vedio as an image.normal returns two parameters.we only use image so used " _, "
    #img = detect(img, faceCascade,eyeCascade,noseCascade,mouthCascade,img_id)
     #check web cam is working
    img = recognize(img,clf,faceCascade)
    cv2.imshow("face detection" , img)  #face detection mean name for the window
    img_id +=1
    if cv2.waitKey(1) & 0xFF == ord('e'):#break the loop if user press q.So terminating condition for loop
        break

video_capture.release()  #release web cam 
cv2.destroyAllWindows() #cv2 destroyed from all the wibdows

