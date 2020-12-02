import os
from flask import Flask, redirect, url_for, request, render_template
from jinja2 import Template
app = Flask(__name__) 


from keras.models import Sequential,load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print("Loading model")
model = tf.keras.models.load_model('BrainTumor.h5')

@app.route('/prediction/<filename>') 
def prediction(filename):
    # Step 1
    img_url = os.path.join('../static/uploads/', filename)
    my_image = plt.imread(os.path.join('static/uploads/', filename))
    # Step 2
    my_image_re = resize(my_image, (128, 128, 1)) 
    # Step 3
    probabilities = model.predict(np.array([my_image_re, ]))[0, :]
    print(probabilities)
    # Step 4
    number_to_class = ["Meningioma","Glioma","Pituitary"]
    index = np.argsort(probabilities)
    predictions = {
            "class1": number_to_class[index[2]],
            "class2": number_to_class[index[1]],
            "class3": number_to_class[index[0]],
            "prob1": probabilities[index[2]],
            "prob2": probabilities[index[1]],
            "prob3": probabilities[index[0]],
        }
    # Step 5
    return render_template('predict2.html', image=img_url, filename=filename)
   
@app.route('/upload',methods = ['POST', 'GET']) 
def upload(): 
     file = request.files['file']
     filename = secure_filename(file.filename)
     file.save(os.path.join('static/uploads/', filename))
     return redirect(url_for('prediction', filename=filename))
    



@app.route('/predict')
def predict():
    return render_template('predict2.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
  
if __name__ == '__main__': 
  app.run(debug = True) 
