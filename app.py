from flask import Flask,render_template,request, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np 
import torch   
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = torch.load("models/model_12_11.py", map_location=torch.device('cpu'))

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
            source = saliency('uploads/' + f.filename, pred)
            print(source)
            return render_template("uploaded.html", name = f.filename, prediction=pred, score = score, source=source) 
    else:
        return "no file uploaded",400 

@app.route('/showMap/saliency/<filename>', methods =["Get","Post"])
def showMap(filename):
    return render_template("map.html", source = filename)

@app.route('/showImg/saliency/<filename>', methods =["Get","Post"])
def showImg(filename):
    return render_template("showImg.html", source=filename)

@app.route('/saliency/maps/<filename>')
def serve_map(filename):
    return send_from_directory("saliency/maps", filename)

@app.route('/saliency/<filename>')
def serve_saliency(filename):
    return send_from_directory("saliency", filename)

def new_img_src():
    pass

def saliency(source,name):
    #we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    #set model in eval mode
    model.eval()
    #transoform input PIL image to torch.Tensor and normalize
    trf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    img = Image.open(source)
    input = trf(img)
    input.unsqueeze_(0)

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input
    input.requires_grad = True
    #forward pass to calculate predictions

    preds = model(input)
    score, indices = torch.max(preds, 1)
    #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

    #apply inverse transform on image
    with torch.no_grad():
        input_img = inv_normalize(input[0])

    slc = slc.cpu().numpy()
    slc = (slc*255).astype(np.uint8)
    in_img = np.transpose(input_img.cpu().detach().numpy(), (1, 2, 0))
    slc_img = slc
    zeros_array1 = np.zeros_like(slc_img)
    zeros_array2 = np.zeros_like(slc_img)
    slc_img = Image.fromarray(np.stack([slc_img, zeros_array1, zeros_array2], axis=-1)).convert("RGB")
    
    blended = Image.blend(img.resize((224,224)), slc_img, 0.5)
    
    if name == "Good":
        output_path = "saliency/good.jpg"
        slc_img.save("saliency/maps/good.jpg", show=False)
    else:
        output_path = "saliency/bad.jpg"
        slc_img.save("saliency/maps/bad.jpg",show=False)
        
    blended.save(output_path,show=False)
    
    return output_path


def preprocess_image(image):
    
    #Preprocess the uploaded image for classification.
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    #image = image.unsqueeze(0)
    return torch.reshape(image,(1,3,224,224))

def postprocess(image):
    trf = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage(),
    ])
    return trf(image)

def classify_image(image):
    
    #Perform classification on the preprocessed image using the loaded model.
    image.requires_grad = True
    prediction = model(image)
    label,score = get_class_label(prediction)
    
    return label, score

def get_score(predicted_class, probabilities):
    
    
    score = probabilities[0, predicted_class].item()
    
    return score * 100
    
def get_class_label(output):
    #Get the class label corresponding to the predicted class index.
    
    class_labels = ["Defective", "Good"]
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    score = get_score(predicted_class, probabilities)
    label = class_labels[predicted_class]
        
    return label, score


if __name__ == '__main__':
    app.run()


