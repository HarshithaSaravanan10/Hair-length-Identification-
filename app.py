# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

# -------------------------
# Load your hair length model
# -------------------------
hair_model = tf.keras.models.load_model("hair_length_vscode.h5")

# -------------------------
# Dummy age & gender function
# Replace this with your actual model if available
# -------------------------
def analyze_age_gender(image_path):
    """
    Returns age and gender prediction.
    Replace this with your actual model if needed.
    Gender is returned as 'Male' or 'Female'.
    """
    # For demo, we randomly generate predictions
    age = np.random.randint(18, 40)
    gender_prob = np.random.rand()
    gender = 'Female' if gender_prob > 0.5 else 'Male'
    return age, gender

# -------------------------
# Hair length prediction
# -------------------------
def predict_hair_length(image):
    """
    Predict hair length using hair_model.
    Returns 'Long Hair' or 'Short Hair'.
    """
    img = image.resize((128,128))  # adjust if your model uses different input size
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = hair_model.predict(img_array)[0][0]
    return "Long Hair" if pred >= 0.5 else "Short Hair"

# -------------------------
# Streamlit GUI
# -------------------------
st.set_page_config(page_title="Long Hair Identification", layout="centered")

st.title("🧑‍🦱 Long Hair Identification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict hair length
    hair_length = predict_hair_length(image)

    # Predict age and gender
    age, gender = analyze_age_gender(uploaded_file)

    # Apply the 20-30 age logic
    if 20 <= age <= 30:
        if hair_length == "Long Hair":
            predicted_gender = "Female"
        else:
            predicted_gender = "Male"
    else:
        predicted_gender = gender

    # Display results
    st.markdown(f"**Age:** {age}")
    st.markdown(f"**Hair Length:** {hair_length}")
    st.markdown(f"**Predicted Gender:** {predicted_gender}")

    # Optional: show image with OpenCV overlay
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    text = f"{hair_length}, {predicted_gender}, Age {age}"
    cv2.putText(img_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Image with Prediction", use_column_width=True)
