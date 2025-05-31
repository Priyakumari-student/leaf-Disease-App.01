import streamlit as st
import numpy as np
from PIL import Image
import json
import base64
import io
import tensorflow as tf

# Disease info data (simplified example, add your full JSON later)
disease_info = {
    "Apple Scab": {
        "prevention": "Avoid wet leaves and use resistant varieties.",
        "cure": "Apply appropriate fungicides."
    },
    "Black Rot": {
        "prevention": "Prune infected parts and keep orchard clean.",
        "cure": "Use recommended fungicides."
    }
    # Add all other disease entries here...
}

# Base64 encoded model string placeholder
model_base64 = """
PASTE_YOUR_FULL_BASE64_MODEL_STRING_HERE
"""

@st.cache(allow_output_mutation=True)
def load_model():
    model_bytes = base64.b64decode(model_base64)
    model_file = io.BytesIO(model_bytes)
    model = tf.keras.models.load_model(model_file)
    return model

model = load_model()

st.title("Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_names = list(disease_info.keys())
    disease_name = class_names[class_idx]
    confidence = prediction[0][class_idx]

    st.write(f"*Prediction:* {disease_name}")
    st.write(f"*Confidence:* {confidence:.2f}")

    st.markdown("### Prevention and Cure Tips")
    st.write(disease_info[disease_name]["prevention"])
    st.write(disease_info[disease_name]["cure"])

    st.markdown("### Feedback")
    rating = st.slider("Rate this app:", 1, 5)
    comment = st.text_area("Leave a comment:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")