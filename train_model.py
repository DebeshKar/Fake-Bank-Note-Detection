import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Feature extractor function (same as in your app)
def extract_features(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((256, 256))  # Resize to fixed size
    img_array = np.array(img).flatten()  # Flatten to 1D array
    entropy = shannon_entropy(img)
    skewness = skew(img_array)
    kurt = kurtosis(img_array)
    variance = np.var(img)
    return [variance, skewness, kurt, entropy]

# Path to the dataset folders
real_images_path = 'data/real/'
fake_images_path = 'data/fake/'

# Collect features and labels
features = []
labels = []

# Function to process the images in a folder and extract features
def process_images(image_folder, label):
    for img_name in os.listdir(image_folder):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_name)
            img = Image.open(img_path)
            feature_vector = extract_features(img)
            features.append(feature_vector)
            labels.append(label)
            print(f"Processed {img_name}")  # Log each processed image

# Process both real and fake images
process_images(real_images_path, 1)  # 1 for real notes
process_images(fake_images_path, 0)  # 0 for fake notes

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model on the training set
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Save the trained model using joblib
import joblib
joblib.dump(model, 'bd_note_model.pkl')
print("Model saved as 'bd_note_model.pkl'")

# Create a DataFrame and save to CSV for further reference
df = pd.DataFrame(features, columns=["Variance", "Skewness", "Kurtosis", "Entropy"])
df['Label'] = labels
df.to_csv('feature_data.csv', index=False)
print("Feature data saved as 'feature_data.csv'")
