import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from imutils import paths
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

"""# converting images into arrays"""

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
         #print(images,Img_labels)
     #print(label)

img=np.array(images)
labels=np.array(Img_labels)

full_x = img
full_y = labels

"""# KAZE buiding"""

import cv2

def get_kaze(images, name='kaze'):
    def get_image_kaze(image, vector_size=32):
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if len(kps) == 0:
            return np.zeros(needed_size)
        
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        
        if dsc.size < needed_size:
            # if we have less than 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return dsc
    
    # KAZE descriptor for all images
    features = []
    for i, img in enumerate(images):
        dsc = get_image_kaze(img)
        features.append(dsc)
    
    result = np.array(features)



"""# Train test and validation split"""

from sklearn.model_selection import train_test_split

X , X_test , y ,y_test = train_test_split(full_x , full_y , test_size=0.2, random_state=0)

train_imgs , val_imgs , y_train , val_y = train_test_split( X, y ,test_size=0.1, random_state=0)

"""# kaze applied"""

kaze_train = get_kaze(train_imgs, name='kaze_train')
kaze_val = get_kaze(val_imgs, name='kaze_val')
kaze_test = get_kaze(X_test, name='kaze_test')

train_imgs=train_imgs.reshape(16145,-1)
val_imgs=val_imgs.reshape(1794,-1)
X_test=X_test.reshape(4485,-1)

print(train_imgs.shape)
print(val_imgs.shape)
print(X_test.shape)

y_train=y_train.reshape(len(y_train),1)
y_test=t_test.reshape(len(t_test),1)
val_y=val_y.reshape(len(val_y),1)

print(y_train.shape)
print(y_test.shape)
print(val_y.shape)
"""# normalisation"""

"""# minmax normalization"""
from sklearn.preprocessing import MinMaxScaler
def normalize(train,val,test):
    min_max_scaler = MinMaxScaler()
    normalize_train_img = min_max_scaler.fit_transform(train)
    normalize_val_img = min_max_scaler.transform(val)
    normalize_test_img = min_max_scaler.transform(test)
    
    return normalize_train_img, normalize_val_img, normalize_test_img

training, validation, testing = normalize(train_imgs, val_imgs, X_test)
print('Normalization done')

# PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.2)
X_train1 = pca.fit_transform(training)
X_test1 = pca.transform(testing)
X_val1 = pca.transform(validation)
print('PCA Applied')

# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train2 = lda.fit_transform(training, y_train)
X_test2 = lda.transform(testing)
X_val2 = lda.transform(validation)
print('LDA Applied')

# SVM for PCA 

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train1, y_train)
print('Training finished')
pca_acc=classifier.score(X_val1,val_y)
print(pca_acc)

# SVM for LDA 

classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(X_train2, y_train)
print('Training finished')
lda_acc = classifier2.score(X_val2,val_y)
print(lda_acc)
