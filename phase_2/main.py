import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Main Model - FacialCNN with modifications
class FacialCNN(nn.Module):
    def __init__(self):
        super(FacialCNN, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)  # Dropout
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 16 * 256, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Forward pass with leaky relu activation
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Variant 1 - Additional Convolutional Layer
class FacialCNNVariant1(nn.Module):
    def __init__(self):
        super(FacialCNNVariant1, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)  # Additional layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2)  # Additional layer
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=5, padding=2) # Additional layer

        # Fully connected layers
        # Adjust the input size of the first fully connected layer based on the output of the last conv layer
        self.fc1 = nn.Linear(1024 * 4 * 4, 128)  # Assuming input images are 64x64
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Variant 2 - Different Kernel Sizes
class FacialCNNVariant2(nn.Module):
    def __init__(self):
        super(FacialCNNVariant2, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)  # Dropout
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 16 * 256, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Forward pass with leaky relu activation
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])  # Adjusted for grayscale (1 channel)
])

# Load and split dataset

train_data = datasets.ImageFolder(r'/Users/roviandsouza/Downloads/train 2/trained_cleaned_images',
                                  transform=transform)
test_data = datasets.ImageFolder(r'/Users/roviandsouza/Downloads/train 2/test_cleaned_images',
                                 transform=transform)
train_idx, val_idx = train_test_split(range(len(train_data)), test_size=0.15)
train_set = Subset(train_data, train_idx)
val_set = Subset(train_data, val_idx)

# Data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


def train_model(model, train_loader, val_loader, epochs=10, early_stopping_patience=5, min_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler
    best_val_loss = np.inf
    early_stopping_counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        # Calculate average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Print training/validation statistics
        print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}')

        # Update learning rate
        scheduler.step()

        # Early stopping logic
      # Check early stopping condition only after min_epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print('Early stopping triggered.')
                break

    return train_loss_history, val_loss_history



def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def evaluate_model(model, test_loader, class_names):
    labels, predictions = get_predictions(model, test_loader)
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, class_names)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(labels, predictions,
                                                                                     average='micro')

    return cm, accuracy, precision, recall, fscore, micro_precision, micro_recall, micro_fscore


# Function to get predictions
def get_predictions(model, loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(train_loss,label="train")
    plt.plot(val_loss,label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():
    # Train and evaluate the main model
    # Train and evaluate the main model with modifications
    model = FacialCNN().to(device)
    model_variant1 = FacialCNNVariant1().to(device)
    model_variant2 = FacialCNNVariant2().to(device)
    class_names = ["Angry", "Bored", "Focused", "Neutral"]
    # train_loss_history, val_loss_history = train_model(model, train_loader, val_loader)
    # torch.save(model.state_dict(), 'facial_cnn_model.pth')
    # plot_loss(train_loss_history, val_loss_history)
    # print("Evaluating Main Model...")
    # metrics = evaluate_model(model, test_loader, class_names)
    # print("Main Model Evaluation Metrics:", metrics)


    train_model(model_variant1, train_loader, val_loader)
    torch.save(model_variant1.state_dict(), 'facial_cnn_variant1_model.pth')  # Save Variant 1
    print("Evaluating Variant 1...")
    metrics_variant1 = evaluate_model(model_variant1, test_loader, class_names)
    print("Variant 1 Evaluation Metrics:", metrics_variant1)

    # Train and evaluate Variant 2
    train_model(model_variant2, train_loader, val_loader)
    torch.save(model_variant2.state_dict(), 'facial_cnn_variant2_model.pth')  # Save Variant 2
    print("Evaluating Variant 2...")
    metrics_variant2 = evaluate_model(model_variant2, test_loader, class_names)
    print("Variant 2 Evaluation Metrics:", metrics_variant2)


if __name__ == "__main__":
    main()
