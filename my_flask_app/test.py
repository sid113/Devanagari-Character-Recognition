import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import numpy as np
from flask import Flask
from flask import Flask,Response , request , flash , url_for,jsonify
import jsonpickle
import os


label_dict={0:"ka",1:"kha",2:"ga",3:"gha",4:"kna",5:"cha",6:"chha",7:"ja",8:"jha",
            9:"yna",10:"ta",11:"thaa",12:"daa",13:"dhaa",14:"adna",15:"tabla(ta)",16:"tha",17:"da",
            18:"dha",19:"na",20:"pa",21:"pha",22:"ba",23:"bha",24:"ma",25:"yaw",26:"ra",
            27:"la",28:"waw",29:"sha",30:"ksha",31:"sa",32:"ha",33:"chhya",34:"tra",35:"gya",
            36:"digit_0",37:"digit_1",38:"digit_2",39:"digit_3",40:"digit_4",41:"digit_5",42:"digit_6",43:"digit_7",44:"digit_8",
            45:"digit_9"
            }
prediction=-1
app=Flask(__name__)
@app.route('/predict',methods=['GET','POST'])
def predict():
    global prediction
    user_file=request.files['file']
    path = os.path.join(os.getcwd()+'/'+user_file.filename)
    user_file.save(path)
    prediction=load_image(path)
    return jsonify({"prediction": label_dict[int(prediction)]})
    
def load_image(filename):
    model = load_model('fix_my_model_1.h5')

    img = cv2.imread(filename,3)
    plt.imshow(img)
    plt.show()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image)
    plt.show()



    width = 32
    height = 32

    dim = (width, height)
 
    # resize image
    resized = cv2.resize(gray_image, dim, interpolation =   cv2.INTER_AREA)
    plt.imshow(resized)
    plt.show()
    img = resized.reshape(1, 32, 32, 1)
    digit = model.predict_classes(img)
    return digit
    
if __name__ == '__main__':
    app.debug=True
    app.run(host='192.168.1.105', port=8089);
