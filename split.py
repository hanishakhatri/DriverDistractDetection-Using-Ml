import pandas as pd
import numpy as np
import cv2
import os
from skimage.transform import resize
from imutils import paths
from sklearn.model_selection import train_test_split
classes = ['c0', 'c1','c2','c3','c4','c5','c6','c7','c8','c9']
Dict = {'c0' : 0, 'c1' :1, 'c2':2, 'c3':3, 'c4':4,'c5':5,'c6':6,'c7':7,'c8':8,'c9':9}
images = []
Img_labels = []
train_path='C:/Users/SAURABH/Desktop/Technocolabs/imgs/train'
for label in classes:
    path = os.path.join(train_path , label)
    print(label)
    for img in os.listdir(path):
        img = cv2.imread(os.path.join(path,img))
        new_img = cv2.resize(img, (64, 64))
        images.append(new_img)
        Img_labels.append(Dict[label])
img=np.array(images)
labels=np.array(Img_labels)
print(img.shape)
print(labels.shape)
x_train,X_test,y_train,Y_test = train_test_split(img,labels, test_size=0.2, random_state=1)
X_train,X_val,Y_train,Y_val = train_test_split(x_train,y_train, test_size=0.1, random_state=1)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(X_val.shape)
print(Y_val.shape)
