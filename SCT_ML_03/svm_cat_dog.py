import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Paths to your single cat and dog images
cat_image_path = r"C:\Users\RUHI MOHAMMED ANANA\Desktop\SCT_ML_03\cat.jpeg"
dog_image_path = r"C:\Users\RUHI MOHAMMED ANANA\Desktop\SCT_ML_03\dog1.jpeg"

def load_single_images(cat_path, dog_path):
    images = []
    labels = []
    
    # Load cat image
    try:
        image = Image.open(cat_path).resize((128, 128))
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            images.append(image)
            labels.append(0)
    except Exception as e:
        print("Error loading cat image:", e)

    # Load dog image
    try:
        image = Image.open(dog_path).resize((128, 128))
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            images.append(image)
            labels.append(1)
    except Exception as e:
        print("Error loading dog image:", e)

    return np.array(images), np.array(labels)

# Load images
images, labels = load_single_images(cat_image_path, dog_image_path)

if images.shape[0] == 2:
    # Flatten for SVM
    X = images.reshape(2, -1)
    y = labels

    # Train on both
    model = SVC(kernel='linear')
    model.fit(X, y)

    # Predict on same images
    preds = model.predict(X)

    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("\nClassification Report:\n", classification_report(y, preds))

    # Visualize
    def visualize_predictions(X, preds, num_images=2):
        plt.figure(figsize=(6, 3))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(X[i].reshape(128, 128, 3))
            plt.title("Predicted: " + ("Dog" if preds[i] == 1 else "Cat"))
            plt.axis('off')
        plt.show()

    visualize_predictions(images, preds)

else:
    print("Both cat and dog images must load correctly!")
