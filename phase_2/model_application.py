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
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Get the name of the predicted class
    predicted_class_name = class_names[predicted.item()]
    return predicted_class_name

# Define class names
class_names = ["Angry", "Bored", "Focused", "Neutral"]

# Example usage
model = load_model('facial_cnn_model.pth', FacialCNN)
prediction = predict_image('/Users/roviandsouza/Downloads/train 2/test_cleaned_images/Neutral/11278.jpg', model)
print("Predicted class:", prediction)
