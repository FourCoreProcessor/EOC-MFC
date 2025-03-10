import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# Load an image (replace 'akka.jpg' with your image path)
image = cv2.imread('man.jpg', cv2.IMREAD_GRAYSCALE)

# Compute HOG features and visualization
fd, hog_image = hog(
    image, 
    orientations=8, 
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1), 
    visualize=True, 
    transform_sqrt=True
)

# Plot the original image, HOG visualization, and HOG histogram
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# HOG image
plt.subplot(1, 3, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.axis('off')

# HOG feature histogram
plt.subplot(1, 3, 3)
plt.hist(fd, bins=20, color='blue', edgecolor='black')
plt.title('HOG Feature Histogram')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()
