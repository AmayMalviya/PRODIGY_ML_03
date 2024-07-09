import zipfile
import numpy as np
import os
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False)

def extract_features_from_zip(zip_path, file_prefix):
    features = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith(file_prefix) and file.endswith(".jpg"):
                label = 0 if 'cat' in file else 1
                with zip_ref.open(file) as image_file:
                    image = Image.open(image_file)
                    image = image.resize((224, 224))
                    image_array = img_to_array(image)
                    image_array = np.expand_dims(image_array, axis=0)
                    image_array = preprocess_input(image_array)
                    
                    features.append(vgg16.predict(image_array).flatten())
                    labels.append(label)
    return np.array(features), np.array(labels)

# Path to the zip file
zip_path = '/Users/amaymalviya/Downloads/dogs-vs-cats/sampleSubmission.csv'  # Update with the actual path

# Extract features and labels from the zip file
cat_features, cat_labels = extract_features_from_zip(zip_path, 'train/cat')
dog_features, dog_labels = extract_features_from_zip(zip_path, 'test/dog')

# Combine features and labels
features = np.vstack((cat_features, dog_features))
labels = np.concatenate((cat_labels, dog_labels))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = svm_model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Classification report
print(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))
