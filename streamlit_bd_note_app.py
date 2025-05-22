import streamlit as st
from PIL import Image
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
import joblib

# Feature extractor from image
def extract_features(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((256, 256))  # Resize to a fixed size
    img_array = np.array(img).flatten()  # Flatten to a 1D array
    entropy = shannon_entropy(img)
    skewness = skew(img_array)
    kurt = kurtosis(img_array)
    variance = np.var(img)
    return [variance, skewness, kurt, entropy]

# Load model
def load_model():
    try:
        model = joblib.load('bd_note_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction function
def predict_note(features):
    model = load_model()
    if model:
        prediction = model.predict([features])[0]
        return "🟢 Real Note" if prediction == 1 else "🔴 Fake Note"
    return "Error in prediction"

# Streamlit UI
st.title("🇧🇩 Bangladeshi Fake Note Detection")
st.write("Upload an image of a banknote to detect if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Upload a note image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Note Image', use_container_width=True)
    with st.spinner('Extracting features and analyzing...'):
        try:
            # Extract features from the image
            features = extract_features(image)

            # Display extracted features
            st.write("### Extracted Features")
            st.write({
                "Variance": round(features[0], 4),
                "Skewness": round(features[1], 4),
                "Kurtosis": round(features[2], 4),
                "Entropy": round(features[3], 4)
            })

            # Get prediction
            prediction = predict_note(features)
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during feature extraction or prediction: {e}. Please check the image format and try again.")
else:
    st.warning("Please upload an image of a banknote.")

