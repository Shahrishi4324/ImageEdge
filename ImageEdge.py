import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/data/img1.jpg')

# Convert the image from BGR to RGB for displaying correctly with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Apply Gaussian blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Display the blurred image
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')
plt.show()

# Perform Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Display the edge-detected image
plt.imshow(edges, cmap='gray')
plt.title('Edge-Detected Image')
plt.axis('off')
plt.show()

# Combine all steps to display original and edge-detected images side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Edge-Detected Image
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge-Detected Image')
plt.axis('off')

plt.show()