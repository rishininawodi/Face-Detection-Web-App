import numpy as np
from PIL import Image
import os,cv2
import numpy as np

#method for train custom classifier to recognize face
def train_classifier(data_dir):
    #read all images in custom data set
    paths = [os.path.join(data_dir,f)for f in os.listdir(data_dir)]
    faces =[]
    ids = []

    #store images in a numpy format and ids of the user on the same index in imageNP and id lists
    for image_path in paths:
        img = Image.open(image_path).convert('L')
        image_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[1].split(".")[1] ) #to get image id

        #append the image
        faces.append(image_np) #image in the number format to the faces 
        #append id
        ids.append(id)
        

        #convert id list in to the number format

        ids= np.array(id)

        #read our classifire theese list and pin it
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        #to save it
        clf.save("classifire.yml")

train_classifier("data")
