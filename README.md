
# 🧠 Brain Tumor Detection Using CNNs

This repository contains a neural network-based project designed to detect brain tumors from MRI images using Convolutional Neural Networks (CNNs). This project automates the detection process, providing a tool that assists radiologists and healthcare professionals in diagnosing brain tumors efficiently.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Frontend](#frontend)
- [References](#references)

## 🧐 Overview
Brain tumor diagnosis traditionally relies on manual inspection of MRI images, which can be time-consuming and error-prone. This project leverages CNNs for automated detection, delivering reliable classification results to streamline the diagnostic process.

## 📂 Dataset
The dataset comprises MRI scans categorized into three classes:
1. **yes** - Images indicating the presence of a brain tumor.
2. **no** - Images with no visible tumor.
3. **pred** - Mix of both classes (for validation after the whole project was ready). 

The data should be organized in a format compatible with common deep learning frameworks, with each class separated into distinct directories for training and testing.

## 🏛 Model Architecture
The model employs a Convolutional Neural Network (CNN) architecture tailored to detect and classify brain tumors in MRI scans. The architecture includes:
- **Convolutional Layers** - For feature extraction
- **Pooling Layers** - To reduce dimensionality
- **Fully Connected Layers** - For classification

This structure is effective in detecting spatial hierarchies in MRI scans, making it well-suited for tumor identification tasks.

## 🛠 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   ```
2. Navigate into the project directory:
   ```bash
   cd brain-tumor-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
1. **Data Preparation**: Ensure your dataset is organized with `train` and `test` folders containing images under `Tumor` and `No Tumor` directories.
2. **Training the Model**: Run the `Training.ipynb` notebook to train the model on your dataset.
3. **Testing and Evaluation**: Use the test dataset to evaluate model performance, tracking metrics like accuracy and recall.
4. **Inference**: Save and load the trained model for rapid MRI scan analysis.

## 📊 Results
Initial tests indicate that the CNN model performs with high accuracy in distinguishing `Tumor` and `No Tumor` images, providing a valuable diagnostic tool. Further tuning and dataset expansion may enhance accuracy.

<img src="https://github.com/user-attachments/assets/d6dda7bb-7297-437b-8b1f-92ee24686e30" alt="Result Screenshot" style="border-radius: 65px;">

## 🌐 Frontend
The project includes a frontend application developed in React to streamline user interaction:
- **Image Upload**: Users can upload MRI images through a drag-and-drop interface using `material-ui-dropzone`.
- **Display of Prediction Results**: After the image is processed, the prediction class and confidence score are displayed.
- **Clear Functionality**: Users can easily reset the display and upload new images.

  
 <img src="https://github.com/user-attachments/assets/d0a43d59-537a-4662-8ce1-bbac22865149" alt="Result Screenshot" width="500" style="border-radius: 35px;">
 <img src="https://github.com/user-attachments/assets/dc6eb399-7dfa-4669-b4bd-ff04526ba5ca" alt="Result Screenshot" width="500" style="border-radius: 35px;">


 *Frontend Interface for MRI Image Upload and Detection*

### Frontend Components
1. **App.js**: Main application component that renders the `ImageUpload` component.
2. **home.js**: Core of the frontend; allows users to upload images, display previews, and see prediction results with confidence scores.
3. **reportWebVitals.js**: Tracks app performance metrics.

