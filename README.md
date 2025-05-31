# SCT_ML_03

TASK: 🐱🐶 Cat vs Dog Image Classification using SVM

This is a simple machine learning project that demonstrates how to classify images of **cats and dogs** using a **Support Vector Machine (SVM)** classifier in Python. The project uses basic image processing and scikit-learn to build and evaluate the model.

## 📌 Objective

To train a binary classification model that can distinguish between images of cats and dogs using individual image files.

## 🛠️ Technologies Used

* Python 🐍
* NumPy
* Pillow (PIL) for image processing
* Scikit-learn (SVM, model evaluation)
* Matplotlib for visualization

## 🧠 Workflow

1. **Image Loading:**
   Load one image each of a cat and a dog from specified file paths.

2. **Preprocessing:**

   * Resize images to 128x128 pixels
   * Convert to RGB and NumPy arrays
   * Flatten images into vectors for model input

3. **Model Training:**
   Train a Support Vector Machine (SVM) with a linear kernel using the preprocessed image data.

4. **Model Evaluation:**

   * Confusion matrix
   * Classification report
   * Visual output of predictions

## 📁 Folder Structure

```
project-folder/
├── cat.jpeg
├── dog1.jpeg
├── svm_cat_dog.py
└── README.md
```

## 📝 How to Run

1. Place your cat and dog images in the same folder as the script.
2. Update the image file paths in the script:

   ```python
   cat_image_path = "cat.jpeg"
   dog_image_path = "dog1.jpeg"
   ```
3. Run the script:

   ```bash
   python svm_cat_dog.py
   ```

## 📷 Sample Output

> Displays prediction results using matplotlib.

## 📌 Note

* This project uses **only one image per class** for demonstration purposes. It is **not suitable for production or real-world accuracy testing**.
* Extend it by adding more labeled images to improve model performance.


