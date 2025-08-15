from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

# Load model and class names
def load_model():
    from fruit_ripeness_classifier import FruitRipenessCNN, datasets_train
    num_classes = len(datasets_train.classes)
    model = FruitRipenessCNN(num_classes)
    model.load_state_dict(torch.load('best_fruit_ripeness_cnn.pth', map_location='cpu'))
    model.eval()
    return model, datasets_train.classes

model, class_names = load_model()

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
def main():
    content = """
    <html>
        <head>
            <title>Fruit Ripeness Classifier</title>
        </head>
        <body>
            <h2>Upload a fruit image to classify ripeness</h2>
            <form action="/predict/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
            </form>
        </body>
    </html>
    """
    return content

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    result = class_names[pred.item()]
    return {"predicted_class": result}
