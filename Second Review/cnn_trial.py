import cv2
import numpy as np
import matplotlib.pyplot as plt

class CNNFeatureExtractor:
    def __init__(self):
        pass

    def conv2d(self, image, kernel):
        kernel = np.flipud(np.fliplr(kernel))
        output = np.zeros((image.shape[0] - 2, image.shape[1] - 2))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)
        return np.maximum(output, 0)  # ReLU Activation

    def max_pooling(self, image, pool_size=(2, 2)):
        # Adjust dimensions to be divisible by the pooling size
        cropped_height = image.shape[0] - (image.shape[0] % pool_size[0])
        cropped_width = image.shape[1] - (image.shape[1] % pool_size[1])
        cropped_image = image[:cropped_height, :cropped_width]

        output = np.zeros((cropped_height // pool_size[0], cropped_width // pool_size[1]))
        for i in range(0, cropped_height, pool_size[0]):
            for j in range(0, cropped_width, pool_size[1]):
                output[i // pool_size[0], j // pool_size[1]] = np.max(cropped_image[i:i+pool_size[0], j:j+pool_size[1]])
        return output.flatten()

    def extract_features(self, image):
        kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        conv1 = self.conv2d(image, kernel1)
        pool1 = self.max_pooling(conv1)
        kernel2 = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
        conv2 = self.conv2d(image, kernel2)
        pool2 = self.max_pooling(conv2)
        return np.concatenate((pool1, pool2)), conv1, conv2

# Load and preprocess the uploaded image
image_path = "man.jpg"  # Path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is resized if it's too large for processing
image = cv2.resize(image, (256, 256))  # Resize to manageable dimensions if necessary

# Instantiate the extractor and extract features
extractor = CNNFeatureExtractor()
features, conv1_output, conv2_output = extractor.extract_features(image)

# Plot the original image and convolution results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(conv1_output, cmap='gray')
plt.title('Convolution 1 Output')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(conv2_output, cmap='gray')
plt.title('Convolution 2 Output')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print the feature vector shape
print("Feature vector shape:", features.shape)
