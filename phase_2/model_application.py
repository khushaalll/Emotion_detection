import torch

from main import FacialCNN

# print(torch.backends.mps.is_available())  # Check if MPS backend is available
# print(torch.backends.mps.is_built())

import torch
from torchvision import transforms
from PIL import Image


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

    return predicted.item()


# Example usage
model = load_model('facial_cnn_model.pth', FacialCNN)
prediction = predict_image('/Users/roviandsouza/Downloads/train 2/trained_cleaned_images/Angry/35190.jpg', model)
print("Predicted class:", prediction)
