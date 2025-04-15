import os
import logging
import cv2
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CNN Model Definition
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flatten_size = 64 * 16 * 8
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x, extract_features=False):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flatten_size)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if extract_features:
            return x
        return self.fc2(x)

# HOG Feature Extraction
def extract_hog_features(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )
    return features

# Image Preprocessing (No Face Detection)
def preprocess_image(image_path):
    image = imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    # Convert to RGB if needed (imread may load as BGR or RGBA)
    if image.shape[-1] == 4:  # RGBA
        image = image[..., :3]
    elif len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    # Resize to 128x64
    image_resized = resize(image, (128, 64), anti_aliasing=True)
    return image_resized

# Test Image Function
def test_image(image_path, cnn_model, svm, scaler, device):
    logger.info(f"Testing image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image {image_path} does not exist.")
        return None
    
    # Preprocess image (no face detection)
    image = preprocess_image(image_path)
    if image is None:
        logger.error("Skipping emotion prediction: Failed to process image.")
        return None
    
    # Preprocess for CNN
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    
    # Extract HOG features
    hog_feat = extract_hog_features(image)
    
    # Extract CNN features
    cnn_model.eval()
    with torch.no_grad():
        cnn_feat = cnn_model(image_tensor, extract_features=True).cpu().numpy()
    
    # Combine and normalize
    combined_feat = np.concatenate([hog_feat, cnn_feat.flatten()])
    logger.info(f"Combined feature shape: {combined_feat.shape}")
    try:
        combined_feat_normalized = scaler.transform([combined_feat])
    except ValueError as e:
        logger.error(f"Scaler transformation failed: {e}")
        return None
    
    # Skip PCA
    logger.info("Skipping PCA for test image...")
    combined_feat_reduced = combined_feat_normalized
    
    # Predict
    try:
        prediction = svm.predict(combined_feat_reduced)
        emotion_classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        logger.info(f"Predicted emotion: {emotion_classes[prediction[0]]}")
        return prediction[0]
    except ValueError as e:
        logger.error(f"SVM prediction failed: {e}")
        return None

# Load and test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Define model paths
    cnn_model_path = 'cnn_model.pth'
    svm_model_path = 'svm_model.pkl'
    scaler_path = 'scaler.pkl'

    # Load CNN model
    if not os.path.exists(cnn_model_path):
        logger.error(f"CNN model file {cnn_model_path} does not exist.")
        exit(1)
    logger.info(f"Loading CNN model from {cnn_model_path}...")
    cnn_model = EmotionCNN(num_classes=7).to(device)
    try:
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        exit(1)
    cnn_model.eval()

    # Load SVM model
    if not os.path.exists(svm_model_path):
        logger.error(f"SVM model file {svm_model_path} does not exist.")
        exit(1)
    logger.info(f"Loading SVM model from {svm_model_path}...")
    try:
        svm = joblib.load(svm_model_path)
    except Exception as e:
        logger.error(f"Failed to load SVM model: {e}")
        exit(1)

    # Load scaler
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file {scaler_path} does not exist.")
        exit(1)
    logger.info(f"Loading scaler from {scaler_path}...")
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        exit(1)

    # Test image
    test_image_path = r"C:\Users\devis\Documents\academic\s2\eocmfc\test24.jpg"
    prediction = test_image(test_image_path, cnn_model, svm, scaler, device)