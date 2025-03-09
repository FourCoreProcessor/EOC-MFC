import numpy as np
import matplotlib.pyplot as plt
import cv2 

# Your HOG function
def get_hog(image):
    gx = np.diff(image, axis=1)  
    gy = np.diff(image, axis=0)  
    magnitude = np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2)  
    orientation = np.arctan2(gy[:, :-1], gx[:-1, :]) * (180 / np.pi) % 180  
    hist_bins = np.histogram(orientation, bins=9, range=(0, 180), weights=magnitude)[0]  
    return hist_bins.flatten()
image = cv2.imread("man.jpg",cv2.IMREAD_GRAYSCALE)
#np.zeros((8, 8))
#for i in range(4):
   # image[i, i+2] = 1  

hog_histogram = get_hog(image)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image (8x8)')
plt.axis('off')
plt.subplot(1, 2, 2)
bins = np.linspace(0, 180, 10)[:-1] 
plt.bar(bins, hog_histogram, width=20, align='edge', color='blue', edgecolor='black')
plt.title('HOG Histogram (9 Bins)')
plt.xlabel('Orientation (degrees)')
plt.ylabel('Magnitude Sum')
plt.xticks(bins)

plt.tight_layout()
plt.show()