from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.feature import hog

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model50.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
        


    def get_hog(images, name='hog', save=False):
        result = np.array([hog(img, block_norm='L2',orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)) for img in images])
        return result
    
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    

    hog_input = get_hog(Xv, name='hog_train', save=True)
    norm_hog_input = min_max_scaler.transform(hog_input)
    pca_norm_hog_train = pca_reload.transform(norm_hog_input)
    model_pca.predict(pca_norm_hog_train)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="safe driving"
    elif preds==1:
        preds="texting - right"
    elif preds==2:
        preds="talking on the phone - right"
    elif preds==3:
        preds="texting - left"
    elif preds==4:
        preds="talking on the phone - left"
    elif preds==5:
        preds="operating the radio"
    elif preds==6:
        preds="drinking"
    elif preds==7:
        preds="reaching behind"
    elif preds==8:
        preds="hair and makeup"
    elif preds==9:
        preds="talking to passenger"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask,redirect,url_for,render_template





# app = Flask(__name__,template_folder='template')


# @app.route("/")
# def home():
#     return render_template("index.html")





# app.run(host="localhost",port=80,debug=True)