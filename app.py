from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import base64

app = Flask(__name__)
dict = {0 : 'Chevrolet', 1 : 'Dodge'}
model = load_model('Resnetmodel.h5')
model.make_predict_function()

def predict(img_path):
    i = image.load_img(img_path, target_size=(100,100))
    
    i = image.img_to_array(i)
    i= cv2.resize(i,(224,224))
    i = np.expand_dims(i, axis=0)
    p = (model.predict(i) > 0.5).astype("int32")
    return dict[p[0][0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict(img_path)
        return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
 #   app.debug = True
    app.run(debug= False)
    
    
