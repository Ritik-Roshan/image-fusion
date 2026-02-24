🧠 Image Fusion Using Convolutional Neural Network (CNN)
📌 Project Overview

Image Fusion is a computer vision technique used to combine relevant information from multiple images into a single enhanced output image.
This project implements Image Fusion using a Convolutional Neural Network (CNN) to merge complementary features from multiple source images while preserving structural and textural information.

The objective of this project is to improve visual quality and information richness by leveraging deep learning–based feature extraction instead of traditional fusion techniques.

This project was developed as part of my University Academic Project in the domain of Deep Learning and Computer Vision.

🎯 Objectives

Perform intelligent fusion of multiple input images.

Extract deep spatial features using CNN architecture.

Preserve edge, texture, and intensity information.

Reduce noise and distortion during fusion.

Generate a high-quality fused image suitable for analysis.

🧱 Project Architecture


Workflow Pipeline

Input Images

     ↓
Preprocessing

     ↓
CNN Feature Extraction

     ↓
Feature Fusion Layer

     ↓
Reconstruction Network

     ↓
Fused Output Image

🧰 Technologies & Tools Used

Category	Tools / Libraries
Programming Language	Python
Deep Learning Framework	TensorFlow / Keras
Image Processing	OpenCV
Numerical Computing	NumPy
Visualization	Matplotlib
Model Training	CNN
Development Environment	Jupyter Notebook
Version Control	Git & GitHub

📂 Project Structure

'''bash
Image-Fusion-CNN/
│
├── dataset/
│   ├── input_images/
│   └── ground_truth/
│
├── models/
│   └── cnn_fusion_model.h5
│
├── notebooks/
│   └── Image_Fusion_CNN.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── fusion.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── fused_images/
│   └── performance_metrics/
│
├── requirements.txt
└── README.md'''

⚙️ Methodology

1️⃣ Data Preprocessing

Image resizing and normalization

Noise reduction

Channel alignment

Conversion into tensor format

Libraries Used:

OpenCV

NumPy

2️⃣ CNN Model Design

The CNN architecture performs automatic feature learning from input images.

Model Components:

Convolution Layers

ReLU Activation

Max Pooling

Feature Map Extraction

Fusion Strategy Layer

Reconstruction Layer

CNN enables:

Edge preservation

Texture enhancement

Spatial feature learning

3️⃣ Feature Fusion Technique

Feature maps extracted from multiple images are combined using:

Weighted averaging

Maximum selection strategy

Deep feature aggregation

This ensures maximum information retention.

4️⃣ Image Reconstruction

The fused feature representation is passed through reconstruction layers to generate the final output image.

🧪 Model Training
Training Parameters
Parameter	Value
Optimizer	Adam
Loss Function	Mean Squared Error (MSE)
Epochs	50
Batch Size	16
Learning Rate	0.001
📊 Evaluation Metrics

Model performance was evaluated using:

Peak Signal-to-Noise Ratio (PSNR)

Structural Similarity Index (SSIM)

Entropy

Mean Squared Error (MSE)

▶️ Installation & Setup

Step 1: Clone Repository

git clone https://github.com/yourusername/Image-Fusion-CNN.git

cd Image-Fusion-CNN

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Run Training

python src/train.py

Step 4: Generate Fused Image

python src/fusion.py

📸 Results

The CNN-based fusion method produces:

Sharper edges

Better contrast

Reduced noise

Enhanced visual clarity

Output images are stored inside:

/results/fused_images

🚀 Applications

Medical Image Fusion (MRI + CT)

Remote Sensing

Surveillance Systems

Multi-focus Photography

Satellite Imaging

Autonomous Systems

🔮 Future Improvements

Attention-based Fusion Networks

GAN-based Image Fusion

Real-time Fusion Deployment

Transformer-based Vision Models

Cloud Deployment
