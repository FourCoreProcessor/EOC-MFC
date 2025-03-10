import numpy as np
import cv2
import os
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_log.log"),
        logging.StreamHandler()
    ]
)

class DataPrep:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_size = (88, 66)
        self.features = []
        self.labels = []
        self.emotion_dict = {"happy": 0, "sad": 1, "anger": 2, "disgust": 3, "neutral": 4,
                             "surprise": 5, "contempt": 6, "fear": 7}

    def preprocess_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image at {image_path}")
            image_resized = cv2.resize(image, self.img_size)
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            return image_gray
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def get_hog(self, image):   
        magnitude, orientation = self.compute_gradients(image)
        hist_bins = self.compute_hog_histogram(magnitude, orientation)
        return hist_bins.flatten()

    def compute_gradients(self, image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi)
        orientation[orientation < 0] += 180
        return magnitude, orientation

    def compute_hog_histogram(self, magnitude, orientation, cell_size=(8, 8), bins=9):
        height, width = magnitude.shape
        cell_h, cell_w = cell_size
        num_cells_x, num_cells_y = width // cell_w, height // cell_h
        histograms = np.zeros((num_cells_y, num_cells_x, bins))
        bin_width = 180 / bins
        
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ori = orientation[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                for y in range(cell_h):
                    for x in range(cell_w):
                        mag, ang = cell_mag[y, x], cell_ori[y, x]
                        bin_idx = min(int(ang // bin_width), bins - 1) if ang < 180 else bins - 1
                        histograms[i, j, bin_idx] += mag
        return histograms.sum(axis=(0, 1))

    def dataset_preprocess(self, cnn_extractor):
        logging.info("Starting dataset preprocessing...")
        for emotion in os.listdir(self.dataset_path):
            emotion_path = os.path.join(self.dataset_path, emotion)
            if os.path.isdir(emotion_path):
                logging.info(f"Processing emotion: {emotion}")
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    try:
                        img_gray = self.preprocess_image(img_path)
                        hog_features = self.get_hog(img_gray)
                        hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-8)
                        cnn_features = cnn_extractor.extract_features(img_gray)
                        cnn_features = cnn_features / (np.linalg.norm(cnn_features) + 1e-8)
                        combined_features = np.concatenate((hog_features, cnn_features))
                        self.features.append(combined_features)
                        self.labels.append(self.emotion_dict[emotion])
                    except Exception as e:
                        logging.warning(f"Skipping image {img_path} due to error: {str(e)}")
        logging.info(f"Preprocessing complete. Total features: {len(self.features)}, Total labels: {len(self.labels)}")
        return np.array(self.features), np.array(self.labels)

class CNNFeatureExtractor:
    def __init__(self):
        pass

    def conv2d(self, image, kernel):
        kernel = np.flipud(np.fliplr(kernel))
        output_h = image.shape[0] - kernel.shape[0] + 1
        output_w = image.shape[1] - kernel.shape[1] + 1
        output = np.zeros((output_h, output_w))
        for i in range(output_h):
            for j in range(output_w):
                output[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)
        return np.maximum(output, 0)  # ReLU Activation

    def max_pooling(self, image, pool_size=(2, 2)):
        h, w = image.shape
        pool_h, pool_w = pool_size
        h_new = (h + pool_h - 1) // pool_h
        w_new = (w + pool_w - 1) // pool_w
        output = np.zeros((h_new, w_new))
        for i in range(h_new):
            for j in range(w_new):
                h_start = i * pool_h
                h_end = min(h_start + pool_h, h)
                w_start = j * pool_w
                w_end = min(w_start + pool_w, w)
                output[i, j] = np.max(image[h_start:h_end, w_start:w_end])
        return output.flatten()

    def extract_features(self, image):
        kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        conv1 = self.conv2d(image, kernel1)
        pool1 = self.max_pooling(conv1)
        kernel2 = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
        conv2 = self.conv2d(image, kernel2)
        pool2 = self.max_pooling(conv2)
        return np.concatenate((pool1, pool2))

class SVMClassifier:
    def __init__(self, gamma=0.5, epochs=1000, learning_rate=0.01, lambda_param=0.01):
        self.gamma = gamma
        self.epochs = epochs
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.classifiers = {}

    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def fit(self, X, y):
        logging.info("Starting SVM training...")
        unique_classes = np.unique(y)
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                y_binary = np.where((y == unique_classes[i]) | (y == unique_classes[j]), 
                                   np.where(y == unique_classes[i], 1, -1), 0)
                mask = y_binary != 0
                logging.info(f"Training binary classifier for classes {unique_classes[i]} vs {unique_classes[j]}")
                self.classifiers[(unique_classes[i], unique_classes[j])] = self.train_binary(
                    X[mask], y_binary[mask])

    def train_binary(self, X, y):
        alphas = np.zeros(len(y))
        bias = 0
        K = np.array([[self.rbf_kernel(x_i, x_j) for x_j in X] for x_i in X])
        for _ in range(self.epochs):
            for i in range(len(y)):
                margin = y[i] * (np.sum(alphas * y * K[i]) + bias)
                if margin >= 1:
                    alphas[i] -= self.lr * (2 * self.lambda_param * alphas[i])
                else:
                    alphas[i] -= self.lr * (2 * self.lambda_param * alphas[i] - y[i])
                alphas[i] = max(0, alphas[i])
        support_vector_indices = np.where(alphas > 1e-5)[0]
        if len(support_vector_indices) > 0:
            bias_sum = 0
            for i in support_vector_indices:
                bias_sum += y[i] - np.sum(alphas * y * K[i])
            bias = bias_sum / len(support_vector_indices)
        return {"alphas": alphas, "bias": bias, "support_vectors": X}

    def predict(self, X):
        from collections import defaultdict
        votes = defaultdict(int)
        for (class1, class2), classifier in self.classifiers.items():
            alphas, bias, support_vectors = classifier.values()
            kernel_matrix = np.array([[self.rbf_kernel(x_i, x_j) for x_j in support_vectors] for x_i in X])
            predictions = np.sign(np.dot(kernel_matrix, alphas) + bias)
            for i, pred in enumerate(predictions):
                predicted_class = class1 if pred > 0 else class2
                votes[predicted_class] += 1
        if votes:
            return max(votes.items(), key=lambda x: x[1])[0]
        return None

class FacialEmotionDetector:
    def __init__(self, dataset_path):
        self.cnn_extractor = CNNFeatureExtractor()
        self.dataset = DataPrep(dataset_path)
        self.svm = SVMClassifier()

    def train_model(self):
        logging.info("Starting model training...")
        try:
            X, y = self.dataset.dataset_preprocess(self.cnn_extractor)
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Training data is empty. Ensure dataset path is correct.")
            self.svm.fit(X, y)
            with open('svm_model.pkl', 'wb') as f:
                pickle.dump(self.svm, f)
            logging.info("Model training completed and saved to svm_model.pkl")
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def predict_emotion(self, image_path):
        logging.info(f"Predicting emotion for image: {image_path}")
        try:
            with open('svm_model.pkl', 'rb') as f:
                self.svm = pickle.load(f)
            img_gray = self.dataset.preprocess_image(image_path)
            hog_feature = self.dataset.get_hog(img_gray)
            cnn_feature = self.cnn_extractor.extract_features(img_gray)
            combined_feature = np.concatenate((hog_feature, cnn_feature))
            prediction = self.svm.predict([combined_feature])
            logging.info(f"Predicted emotion: {prediction}")
            return prediction
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

if __name__ == "__main__":
    dataset_path = "dataset"  # Replace with your dataset path
    detector = FacialEmotionDetector(dataset_path)
    detector.train_model()  # Train the model and save it