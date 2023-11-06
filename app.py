from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import os
import numpy as np 
import torch   
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = torch.load("models/model.py", map_location=torch.device('cpu'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    
    if 'file' in request.files:
        if request.method == 'POST':
            f = request.files['file']
            if f.name == '':
                return "no file selected",400
            img_path =os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
            f.save(img_path)
            image = Image.open('uploads/' + f.filename)
            image = preprocess_image(image) 
            pred, score = classify_image(image)
            
            return render_template("uploaded.html", name = f.filename, prediction=pred, score = score) 
    else:
        return "no file uploaded",400 


def preprocess_image(image):
    
    #Preprocess the uploaded image for classification.
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),])
    image = transform(image)
    return torch.reshape(image,(1,3,224,224))

def classify_image(image):
    
    #Perform classification on the preprocessed image using the loaded model.
    
    prediction = model(image)
    score = get_score(prediction.item())
    class_index = torch.round(prediction.data).int()
    class_label = get_class_label(class_index)

    return class_label, score

def get_score(prediction):
    
    if prediction < 0.5:
        result = "{:.10f}".format(100*(1-prediction))
    else: 
        result = "{:.10f}".format(100*prediction)
        
    return result[:5] 
    
def get_class_label(class_index):
    
    #Get the class label corresponding to the predicted class index.
    
    # Define your class labels here
    class_labels = ['Defective', 'Good']

    return class_labels[class_index]

if __name__ == '__main__':
    app.run()


"""
from flask import Flask, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('path_to_model')

@app.route('/upload', methods=['POST'])
def upload_image():
    
    #Function to upload an image for classification.
    
    if 'image' not in request.files:
        return 'No image file found', 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image = preprocess_image(image)

    # Perform classification
    result = classify_image(image)

    return result

def preprocess_image(image):
    
    #Preprocess the uploaded image for classification.
    
    image = image.resize((224, 224))  # Resize the image to match the input size of the model
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize pixel values to range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image

def classify_image(image):
    
    #Perform classification on the preprocessed image using the loaded model.
    
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = get_class_label(class_index)

    return class_label

def get_class_label(class_index):
    
    #Get the class label corresponding to the predicted class index.
    
    # Define your class labels here
    class_labels = ['class1', 'class2', 'class3']

    return class_labels[class_index]

if __name__ == '__main__':
    app.run()

    
"""