# AAI_Emotion_detection

## Description

This project focuses on emotion detection from images. With the use of different image processing techniques and machine learning, we strive to accurately identify emotional states like Angry, Bored, Focused, and Neutral from image datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributers](#contributers)
- [Datasets](#datasets)

## Installation

### Prerequisites:

1. **Python**: Ensure you have a recent version of Python installed. This project was developed with Python 3, so it's recommended to use Python 3.x.
2. **pip**: Ensure that you have the `pip` package installer for Python. This will be used to install the required libraries.

### Dependencies:

This project utilizes several Python libraries. Here's how to install them:

- **OpenCV**: This library provides tools for image processing and computer vision tasks.
pip install opencv-python

- **Scikit-learn**: Used for machine learning tasks, especially for the `train_test_split` function.
pip install scikit-learn

- **NumPy**: A library for numerical operations in Python.
pip install numpy

- **Scikit-image (skimage)**: This library offers a collection of algorithms for image processing.
pip install scikit-image

- **Matplotlib**: A plotting library to visualize data and results.
pip install matplotlib

### Steps:

1. **Clone the Repository**: First, clone this repository to your local machine using `git`.
git clone [https://github.com/khushaalll/AAI_Emotion_detection/tree/main]

3. **Install Dependencies**: Run the commands provided above to install all the required libraries.
4. **Prepare the Data**: Make sure you have the image datasets placed in the appropriate directories (`train_dir` and `test_dir` as specified in the code).
5. **Run the Script**: Execute the Python script to preprocess images, visualize the data, or any other tasks as defined within.

## Usage

Once set up, use the provided scripts to preprocess images, train the model, and evaluate its performance. For detailed usage, refer to the inline comments in the scripts.

## Features

- Image resizing to a consistent dimension of 256x256 pixels.
- Conversion of images to grayscale to eliminate color inconsistencies and emphasize facial expressions.
- Preprocessed datasets with cleaned and standardized images.
- Visualization tools to understand class distributions and pixel intensity distributions in the dataset.

## Contributors

 
- Darshil Patil - Data Specialist 
- Khushal Jain - Training Specialist
- Rovian Dsouza - Evaluation Specialist


## Datasets:
The datasets used are uploaded in the repo. The below links are sources of the datasets.
- **Training Data**: [Link to Training Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
- **Additional Data**: [Link to Additional Dataset](https://universe.roboflow.com/university-ggw0y/emotional-detection-r3xb2)
