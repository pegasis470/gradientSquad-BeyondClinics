from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import keras
from flask_cors import CORS
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from keras import models
from keras.layers import DepthwiseConv2D #type:ignore
from keras.utils import custom_object_scope #type:ignore
app = Flask(__name__)
CORS(app)
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)
    return DepthwiseConv2D(*args, **kwargs)

def process_image_and_classify(image,req_type):
    # Convert image to grayscale numpy array
    if req_type == "brest_cancer":
        with custom_object_scope({'DepthwiseConv2D': custom_depthwise_conv2d}):
            model = models.load_model('keras_model-4.h5')
        class_names = open("labels-breast.txt", "r").readlines()
        class_indexes={}
        for i in class_names:
            i=i.strip('\n').split(' ')
            class_indexes[int(i[0])]=i[-1]
    elif req_type == "throat":
        with custom_object_scope({'DepthwiseConv2D': custom_depthwise_conv2d}):
            model = models.load_model('keras_model.h5')
        class_names = open("labels.txt", "r").readlines()
        class_indexes={}
        for i in class_names:
            i=i.strip('\n').split(' ')
            class_indexes[int(i[0])]=i[-1]
    elif req_type == "brain":
        with custom_object_scope({'DepthwiseConv2D': custom_depthwise_conv2d}):
            model = models.load_model('keras_model-brain.h5',compile=True   )
        class_names = open("labels-brain.txt", "r").readlines()
        class_indexes={}
        for i in class_names:
            i=i.strip('\n').split(' ')
            class_indexes[int(i[0])]=i[-1]  
    else:
        return jsonify({"error":"invalid request type"})
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_indexes.get(index)
    confidence_score = prediction[0][index]
    return class_name,confidence_score*100

@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    req=request.form.get("label")
    print(req)
    try:
        # Open the image file
        file.save(f"uploads/{filename}")
        image = Image.open(f"uploads/{filename}").convert("RGB")
        # Process the image and classify it
        label,score = process_image_and_classify(image,req)
        return jsonify({"label": label,"score":score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)