import torch
from torchvision import transforms
from PIL import Image

# Assuming FacialCNN is defined in main.py
from main import FacialCNN

def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    # prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted.item()]
    return predicted_class_name

class_names = ["Angry", "Bored", "Focused", "Neutral"]

model = load_model('facial_cnn_model.pth', FacialCNN)
prediction = predict_image('/Users/roviandsouza/Downloads/train 2/test_cleaned_images/Bored/11192.jpg', model)
print("Predicted class:", prediction)
