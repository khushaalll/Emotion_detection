import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Assuming FacialCNN and other necessary functions are defined in main.py
from main import FacialCNN, plot_confusion_matrix, get_predictions, FacialCNNVariant1

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model_path, model_class, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image_path, model, transform, device):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Ensure the image is on the correct device

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted.item()]
    return predicted_class_name

def evaluate_model(model, test_loader, class_names):
    labels, predictions = get_predictions(model, test_loader)
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, class_names)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(labels, predictions,
                                                                                     average='micro')

    return cm, accuracy, precision, recall, fscore, micro_precision, micro_recall, micro_fscore

def prepare_bias_test_data(root_dir, transform):
    test_data_gender_1 = datasets.ImageFolder(root_dir + 'test_gender/Women', transform=transform)
    test_data_gender_2 = datasets.ImageFolder(root_dir + 'test_gender/Men_test_cleaned_images', transform=transform)
    test_data_age_group_1 = datasets.ImageFolder(root_dir + 'test_age/young', transform=transform)
    test_data_age_group_2 = datasets.ImageFolder(root_dir + 'test_age/middle', transform=transform)
    test_data_age_group_3 = datasets.ImageFolder(root_dir + 'test_age/old', transform=transform)
    org_test_data = datasets.ImageFolder('/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/test_cleaned_images', transform=transform)
    org_train_data = datasets.ImageFolder('/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/trained_cleaned_images', transform=transform)

    test_loader_gender_1 = DataLoader(test_data_gender_1, batch_size=32, shuffle=False)
    test_loader_gender_2 = DataLoader(test_data_gender_2, batch_size=32, shuffle=False)
    test_loader_age_group_1 = DataLoader(test_data_age_group_1, batch_size=32, shuffle=False)
    test_loader_age_group_2 = DataLoader(test_data_age_group_2, batch_size=32, shuffle=False)
    test_loader_age_group_3 = DataLoader(test_data_age_group_3, batch_size=32, shuffle=False)
    test_loader_org_test = DataLoader(org_test_data, batch_size=32, shuffle=False)
    test_loader_org_trained = DataLoader(org_train_data, batch_size=32, shuffle=False)

    return {
        "Gender: Women: ": test_loader_gender_1,
        "Gender: Men: ": test_loader_gender_2,
        "Young age: ": test_loader_age_group_1,
        "Middle age: ": test_loader_age_group_2,
        "Old age: ": test_loader_age_group_3,
        "org test data":test_loader_org_test,
        "org train data": test_loader_org_test
    }


def evaluate_bias(model, loaders, class_names):
    bias_results = {}
    for key, loader in loaders.items():
        print(f"Evaluating on {key}...")
        metrics = evaluate_model(model, loader, class_names)
        bias_results[key] = metrics
        print(f"{key} Evaluation Metrics:", metrics)
    return bias_results


class_names = ["Angry", "Bored", "Focused", "Neutral"]
model = load_model(r'/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/phase_3/facial_cnn_model.pth', FacialCNN, device)


# Image prediction example
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Evaluate bias
root_dir = '/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/phase_3/'
test_loaders = prepare_bias_test_data(root_dir, transform)
bias_results = evaluate_bias(model, test_loaders, class_names)
