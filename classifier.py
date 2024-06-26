import numpy as np
from PIL import Image
import os,cv2

def train_classfier(data_dir):
    path = [os.path.join(data_dir,f)for f in os.listdir(data_dir)]
    faces = {}
    ids = {}
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1] ) #to get image id

        #append the image
        faces.append(imageNp) #image in the number format to the faces 
        #append id
        ids.append(id)

        #convert id list in to the number format

        ids= np.array(ids)

        #read our classifire theese list and pin it
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        #to save it
        clf.write("classifire.yml")

train_classfier("data")
