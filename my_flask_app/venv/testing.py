from flask import flask
from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt;
model = load_model('my_model.h5')
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
import csv

app=Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    filename=request.files['data'];
    image=load_image(filename);
    prediction=run_model(image)[0]
    
def load_image(filename):    
    img = cv2.imread('untitled.png',3)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = 28
    height = 28
    dim = (width, height)
 
    # resize image
    resized = cv2.resize(gray_image, dim, interpolation = cv2.INTER_AREA)
    return resized

def run_model(image):
    model = load_model('my_model.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
    digit = model.predict_classes(img)
    return digit[0];

