import numpy as np
import torch
import cv2
import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from lib.model import RiceYieldCNN  # Your model definition for yield prediction

# Classification function for texture analysis
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

# Function to open file dialog and load the image for texture analysis
def open_file_for_texture():
    filename = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")))
    if filename:
        process_image_for_texture(filename)

# Function to process the image for texture analysis
def process_image_for_texture(i):
    print("Rice Texture analyser by Balaji M ,Ajai S & Dinesh Kumar")
    
    img = cv2.imread(i, 0)  # Read the image in grayscale

    # Convert to binary
    ret, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

    # Apply an averaging filter
    kernel = np.ones((5, 5), np.float32) / 9
    dst = cv2.filter2D(binary, -1, kernel)

    # Structuring element for morphological operations
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Erosion and Dilation
    erosion = cv2.erode(dst, kernel2, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)

    # Edge detection
    edges = cv2.Canny(dilation, 100, 200)

    # Size detection (aspect ratio of rice grains)
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
    avg_ar = total_ar / len(contours) if contours else 0
    print("Average Aspect Ratio=", round(avg_ar, 2), get_classification(avg_ar))

    # Create the plot for texture analysis
    plot_images_for_texture(img, binary, dst, erosion, dilation, edges)

# Function to create and display texture analysis plots
def plot_images_for_texture(img, binary, dst, erosion, dilation, edges):
    imgs_row = 2
    imgs_col = 3
    plt.figure(figsize=(10, 8))

    plt.subplot(imgs_row, imgs_col, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(imgs_row, imgs_col, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")

    plt.subplot(imgs_row, imgs_col, 3)
    plt.imshow(dst, cmap='gray')
    plt.title("Filtered Image")

    plt.subplot(imgs_row, imgs_col, 4)
    plt.imshow(erosion, cmap='gray')
    plt.title("Eroded Image")

    plt.subplot(imgs_row, imgs_col, 5)
    plt.imshow(dilation, cmap='gray')
    plt.title("Dilated Image")

    plt.subplot(imgs_row, imgs_col, 6)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")

    plt.tight_layout()
    plt.show()

# Define the yield prediction function
def predict_yield(checkpoint_path, image_dir, save_csv=False):
    """
    Predict rice yield based on images in the specified directory using a pre-trained model.
    
    :param checkpoint_path: Path to the model checkpoint (.pth file)
    :param image_dir: Directory containing images to be processed
    :param save_csv: Boolean flag to save results to a CSV file
    :return: None
    """
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define input resolution and normalization parameters
    input_resolution = (512, 512)
    mean = 0.5
    std = 0.5

    # Get list of image paths in the specified directory
    image_path_list = sorted(glob(os.path.join(image_dir, "*")))

    # Load the pre-trained model
    model = RiceYieldCNN()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    results = []  # List to store results if CSV output is requested

    print(" ")
    print("==================================================")
    print("Rice Yield Predictor by Balaji , Ajai & Dinesh kumar")
    print("==================================================")
    print(" ")
    
    # Loop through each image and predict yield
    for i, image_path in enumerate(image_path_list):
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_resolution)

        # Normalize the image
        input_img = image.astype(np.float32) / 255.0
        input_img = (input_img - np.array(mean).astype(np.float32)) / \
                    np.array(std).astype(np.float32)
        input_img = input_img.transpose(2, 0, 1)
        input_img = torch.Tensor(input_img).unsqueeze(0).to(device)

        # Get model prediction
        pred_yield = model(input_img)
        pred_yield = round(float(pred_yield.squeeze(0).detach().cpu().numpy()), 2)

        # Print prediction
        print(f"{image_name}: {pred_yield} g/m2, {round(pred_yield / 100, 2)} t/ha")

        if save_csv:
            # Append results to list
            results.append({
                "id": i,
                "image_name": image_name,
                "gpms": pred_yield,
                "tpha": pred_yield / 100
            })

    # Save results as CSV if requested
    if save_csv:
        pd.DataFrame.from_records(results).to_csv("out.csv", index=False)

    # Generate plots if needed
    if save_csv:
        df = pd.DataFrame.from_records(results)
        plt.figure(figsize=(10, 6))
        plt.bar(df['image_name'], df['gpms'], color='skyblue')
        plt.xlabel('Image')
        plt.ylabel('Predicted Yield (g/m2)')
        plt.title('Predicted Rice Yield for Each Image')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# GUI Setup using Tkinter
def create_gui():
    root = tk.Tk()
    root.title("Rice Yield & Texture Analyzer")

    # Set the window size
    root.geometry("400x300")

    # Add Yield Prediction Button
    yield_button = tk.Button(root, text="Yield Prediction", width=20, height=2, command=yield_prediction)
    yield_button.pack(pady=20)

    # Add Texture Analysis Button
    texture_button = tk.Button(root, text="Texture Analysis", width=20, height=2, command=open_file_for_texture)
    texture_button.pack(pady=20)

    # Start the Tkinter event loop
    root.mainloop()

# Function to handle Yield Prediction
def yield_prediction():
    # Ask user for the image folder and checkpoint
    image_dir = filedialog.askdirectory(title="Select Image Folder for Yield Prediction")
    if not image_dir:
        return

    checkpoint_path = filedialog.askopenfilename(
        title="Select the Model Checkpoint", filetypes=[("PT Files", "*.pth")]
    )
    if not checkpoint_path:
        return

    # Run the prediction
    predict_yield(checkpoint_path=checkpoint_path, image_dir=image_dir, save_csv=True)
    print("Yield prediction completed!")

# Run the GUI
if __name__ == "__main__":
    create_gui()
