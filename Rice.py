# rice_app.py
import os
import cv2
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import gdown
import zipfile
import os
import streamlit as st

@st.cache_resource
def download_and_extract_data():
    if not os.path.exists("Rice_Image_Dataset"):
        file_id = "1FxW2XuLAO9khbaU0RBB_UN1g7WvTIDv1"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "Rice_Image_Dataset.zip"

        # Try downloading the zip with fuzzy=True to bypass Google Drive warnings
        gdown.download(url, output, quiet=False, fuzzy=True)

        # Extract it
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()

        os.remove(output)


download_and_extract_data()

# Configuration
IMAGE_SIZE = 32
PCA_COMPONENTS = 15
MAX_IMAGES_PER_CLASS = 300
CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
DATASET_DIR = 'Rice_Image_Dataset'

# Load and preprocess images
@st.cache_data
def load_dataset():
    X, y = [], []
    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        files = os.listdir(class_dir)[:MAX_IMAGES_PER_CLASS]
        for file in files:
            path = os.path.join(class_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).flatten()
                X.append(img)
                y.append(idx)
    return np.array(X), np.array(y)

@st.cache_resource
def train_model(X, y):
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=50, n_jobs=-1, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, pca, acc

def measure_grain_quality(gray_img):
    """Estimate basic grain quality features: length, width, aspect ratio"""
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        length = max(w, h)
        width = min(w, h)
        aspect_ratio = round(length / width, 2) if width > 0 else 0
        return length, width, aspect_ratio
    else:
        return None, None, None

# Streamlit App
st.title("🌾 Rice Grain Classifier + Quality Estimator")
st.markdown("Classifies rice grain type and measures basic quality parameters (length, width, aspect ratio)")

with st.spinner("🔄 Loading and training..."):
    X, y = load_dataset()
    model, pca, acc = train_model(X, y)
    st.success(f"✅ Model trained with test accuracy: {acc*100:.2f}%")

# Image Upload
uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).flatten()
    img_pca = pca.transform([img_resized])
    pred = model.predict(img_pca)[0]

    st.image(cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_GRAY2RGB), caption="Uploaded Image", width=150)
    st.success(f"🌾 Predicted Rice Type: **{CLASSES[pred]}**")

    # Measure and display quality features
    length, width, aspect = measure_grain_quality(img)
    if length is not None:
        st.markdown("**Estimated Quality Parameters:**")
        st.markdown(f"- **Length:** {length} px")
        st.markdown(f"- **Width:** {width} px")
        st.markdown(f"- **Aspect Ratio:** {aspect}")
    else:
        st.warning("Could not detect grain contour. Please upload a clearer image.")
