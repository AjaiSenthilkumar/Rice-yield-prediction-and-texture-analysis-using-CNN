import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_classification(ratio):
    ratio = round(ratio, 1)
    to_ret = ""
    if ratio >= 3:
        to_ret = "Slender"
    elif 2.1 <= ratio < 3:
        to_ret = "Medium"
    elif 1.1 <= ratio < 2.1:
        to_ret = "Bold"
    elif ratio <= 1:
        to_ret = "Round"
    to_ret = "(" + to_ret + ")"
    return to_ret

print("Rice Texture analyser by Sowmiya")
i = input("Enter the Image with extension to be processed:")
img = cv2.imread(i, 0)  

# convert into binary
ret, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)  

# averaging filter
kernel = np.ones((5, 5), np.float32) / 9
dst = cv2.filter2D(binary, -1, kernel)  # -1 : depth of the destination image

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# erosion
erosion = cv2.erode(dst, kernel2, iterations=1)

# dilation
dilation = cv2.dilate(erosion, kernel2, iterations=1)

# edge detection
edges = cv2.Canny(dilation, 100, 200)

# Size detection
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("No. of rice grains=", len(contours))
total_ar = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    if aspect_ratio < 1:
        aspect_ratio = 1 / aspect_ratio
    print(round(aspect_ratio, 2), get_classification(aspect_ratio))
    total_ar += aspect_ratio
avg_ar = total_ar / len(contours)
print("Average Aspect Ratio=", round(avg_ar, 2), get_classification(avg_ar))

# plot the images
imgs_row = 2
imgs_col = 3
plt.subplot(imgs_row, imgs_col, 1), plt.imshow(img, 'gray')
plt.title("Original image")

plt.subplot(imgs_row, imgs_col, 2), plt.imshow(binary, 'gray')
plt.title("Binary image")

plt.subplot(imgs_row, imgs_col, 3), plt.imshow(dst, 'gray')
plt.title("Filtered image")

plt.subplot(imgs_row, imgs_col, 4), plt.imshow(erosion, 'gray')
plt.title("Eroded image")

plt.subplot(imgs_row, imgs_col, 5), plt.imshow(dilation, 'gray')
plt.title("Dilated image")

plt.subplot(imgs_row, imgs_col, 6), plt.imshow(edges, 'gray')
plt.title("Edge detect")

plt.show()
