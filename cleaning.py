import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np
from skimage import io
train_dir = r"/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/train"
test_dir = r"/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/test"
labels = ["Angry", "Bored", "Focused", "Neutral"]
train_target_dir = r"/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/trained_cleaned_images"
test_target_dir = r"/Users/roviandsouza/Documents/GitHub/AAI_Emotion_detection/test_cleaned_images"
x_train = []
y_train = []
x_test = []
y_test = []
for label in labels:
    os.makedirs(os.path.join(train_target_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_target_dir, label), exist_ok=True)

# Cleaning in train folder
for label, class_label in enumerate(labels):
    class_path = os.path.join(train_dir, class_label)
    for file in os.listdir(class_path):
        image_path = os.path.join(class_path, file)
        image = cv2.imread(image_path)
        resized_img = cv2.resize(image, (256, 256))
        # GrayScale
        image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        io.imsave(os.path.join(train_target_dir, class_label, file), image)
        x_train.append(image)
        y_train.append(label)

# Cleaning in test folder
for label, class_label in enumerate(labels):
    class_path = os.path.join(test_dir, class_label)
    for file in os.listdir(class_path):
        image_path = os.path.join(class_path, file)

        image = cv2.imread(image_path)
        resized_img = cv2.resize(image,(256, 256))
        # GrayScale
        image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        io.imsave(os.path.join(test_target_dir,class_label,file), image)
        x_test.append(image)
        y_test.append(label)


# Total images
class_counts = np.bincount(y_train)
plt.figure(figsize=(8, 6))
plt.bar(labels, class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution (Training Set)')
plt.show()


sample_indices = np.random.choice(len(x_train), 25, replace=False)
plt.figure(figsize=(10, 10))
for i, index in enumerate(sample_indices):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[index])
    plt.title(f'Class {labels[y_train[index]]}')
    plt.axis('off')
plt.suptitle('Sample Images', fontsize=16)
plt.show()

# Step 5: Pixel Intensity Distribution Visualization Histogram
plt.figure(figsize=(12, 6))
for i, index in enumerate(sample_indices):
    plt.subplot(5, 5, i + 1)
    plt.hist(x_train[index].ravel(), bins=20, color='blue', alpha=0.7)
    plt.title(f'Class {labels[y_train[index]]}')
    plt.xlabel('Pixel Intensity')
plt.suptitle('Pixel Intensity Distribution', fontsize=16)
plt.tight_layout()
plt.show()