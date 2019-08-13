from django.shortcuts import render
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
global graph,model

#initializing the graph
graph = tf.get_default_graph()

#loading our trained model
print("Keras model loading.......")
model = load_model('detection_app/dmg_car-weights-CNN.h5')
print("Model loaded!!")


def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']
        img = image.load_img(myfile, target_size=(64,64,3))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        with graph.as_default():
            preds = model.predict(img)
        if preds[0][0] >= 0.7:
            result = 'The Car is not Damaged'
        else:
            result = 'The Car is Damaged'
        return render(request, "detection_app/pred.html", {
            'result': result})
    else:
        return render(request, "detection_app/pred.html")

