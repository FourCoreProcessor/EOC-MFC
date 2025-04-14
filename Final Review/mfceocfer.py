import os
import logging
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define log format
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

# File handler
log_file = 'training.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

# CNN Model Definition
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x64 -> 64x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x32 -> 32x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x16 -> 16x8
        self.dropout = nn.Dropout(0.5)
        self.flatten_size = 64 * 16 * 8
        self.fc1 = nn.Linear(self.flatten_size, 256)  # Feature vector
        self.fc2 = nn.Linear(256, num_classes)  # Classification head
    
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

# Custom Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = imread(self.image_paths[idx])
            image = resize(image, (128, 64), anti_aliasing=True)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            return None, None

# HOG Feature Extraction
def extract_hog_features(image):
    image = resize(image, (128, 64), anti_aliasing=True)  # Ensure consistent size
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )
    return features

# CNN Feature Extraction
def extract_cnn_features(dataloader, model, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if images is None:
                continue
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            features = model(images, extract_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

# Train CNN with Early Stopping
def train_cnn(model, train_loader, val_loader, device, num_epochs=20, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None
    
    model.train()
    logger.info("Starting CNN training with early stopping...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            if images is None:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
            logger.info("New best validation loss, saving weights...")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    model.load_state_dict(best_weights)
    logger.info("CNN training completed with best weights.")
    return model

# Main Function
def main(dataset_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    emotion_classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    image_paths = []
    labels = []
    
    logger.info("Loading dataset...")
    for idx, emotion in enumerate(emotion_classes):
        emotion_dir = os.path.join(dataset_root, emotion)
        if not os.path.exists(emotion_dir):
            logger.warning(f"Directory {emotion_dir} not found, skipping.")
            continue
        img_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))]
        logger.info(f"Found {len(img_files)} images for {emotion}")
        image_paths.extend(os.path.join(emotion_dir, f) for f in img_files)
        labels.extend([idx] * len(img_files))
    
    if not image_paths:
        logger.error("No images found in dataset!")
        return None, None, None
    
    train_idx, val_idx = train_test_split(range(len(image_paths)), test_size=0.2, stratify=labels, random_state=42)
    full_dataset = EmotionDataset(image_paths, labels, transform=transform)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    cnn_model_path = 'cnn_model.pth'
    cnn_model = EmotionCNN(num_classes=7).to(device)
    if os.path.exists(cnn_model_path):
        logger.info(f"Loading existing CNN model from {cnn_model_path}")
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    else:
        logger.info("Training new CNN model...")
        cnn_model = train_cnn(cnn_model, train_loader, val_loader, device, num_epochs=20, patience=3)
        torch.save(cnn_model.state_dict(), cnn_model_path)
        logger.info(f"Best CNN model saved as {cnn_model_path}")
    
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=0)
    logger.info("Extracting CNN features...")
    cnn_features, cnn_labels = extract_cnn_features(full_loader, cnn_model, device)
    
    logger.info("Extracting HOG features...")
    hog_features = np.array([extract_hog_features(imread(path)) for path in image_paths])
    
    logger.info("Combining HOG and CNN features...")
    combined_features = np.concatenate([hog_features, cnn_features], axis=1)
    logger.info(f"Combined features shape: {combined_features.shape}")
    
    scaler_path = 'scaler.pkl'
    scaler = StandardScaler()
    if os.path.exists(scaler_path):
        logger.info(f"Loading existing scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        combined_features_normalized = scaler.transform(combined_features)
    else:
        logger.info("Fitting new scaler...")
        combined_features_normalized = scaler.fit_transform(combined_features)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Feature scaler saved as {scaler_path}")
    
    logger.info("Skipping PCA, using normalized combined features directly...")
    combined_features_reduced = combined_features_normalized
    logger.info(f"Feature shape: {combined_features_reduced.shape}")
    
    svm_model_path = 'svm_model.pkl'
    if os.path.exists(svm_model_path):
        logger.info(f"Loading existing SVM model from {svm_model_path}")
        svm = joblib.load(svm_model_path)
    else:
        # Skip redundancy check to save time
        logger.info("Training SVM with OvO and hyperparameter tuning...")
        param_grid = {'estimator__C': [0.1, 1, 10], 'estimator__gamma': ['scale', 'auto']}
        svm_base = OneVsOneClassifier(SVC(kernel='rbf', cache_size=1000))
        grid_search = GridSearchCV(svm_base, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(combined_features_reduced, labels)
        svm = grid_search.best_estimator_
        logger.info(f"Best SVM parameters: {grid_search.best_params_}")
        joblib.dump(svm, svm_model_path)
        logger.info(f"SVM model saved as {svm_model_path}")
    
    logger.info("Computing confusion matrix...")
    predictions = svm.predict(combined_features_reduced)
    cm = confusion_matrix(labels, predictions)
    logger.info("Confusion Matrix:\n" + str(cm))
    
    return cnn_model, svm, scaler

# Test Function
def test_image(image_path, cnn_model, svm, scaler, device):
    logger.info(f"Testing image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image {image_path} does not exist.")
        return None
    
    image = imread(image_path)
    image_resized = resize(image, (128, 64), anti_aliasing=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image_resized).unsqueeze(0).to(device, dtype=torch.float32)
    
    hog_feat = extract_hog_features(image_resized)
    
    cnn_model.eval()
    with torch.no_grad():
        cnn_feat = cnn_model(image_tensor, extract_features=True).cpu().numpy()
    
    combined_feat = np.concatenate([hog_feat, cnn_feat.flatten()])
    logger.info(f"Test combined features shape: {combined_feat.shape}")
    try:
        combined_feat_normalized = scaler.transform([combined_feat])
    except ValueError as e:
        logger.error(f"Scaler transformation failed: {e}")
        return None
    
    combined_feat_reduced = combined_feat_normalized
    logger.info("Skipping PCA for test image...")
    
    try:
        prediction = svm.predict(combined_feat_reduced)
        emotion_classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        logger.info(f"Predicted emotion: {emotion_classes[prediction[0]]}")
        return prediction[0]
    except ValueError as e:
        logger.error(f"SVM prediction failed: {e}")
        return None

if __name__ == "__main__":
    dataset_root = r"C:\Users\devis\Documents\python\dataset"
    cnn_model, svm, scaler = main(dataset_root)
    
    if cnn_model is not None:
        test_image_path = r"C:\Users\devis\Documents\academic\s2\eocmfc\test1.jpg"
        prediction = test_image(test_image_path, cnn_model, svm, scaler, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))