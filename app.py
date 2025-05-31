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
<start of Base64 string>
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
pErCNFVrkYqa7NzKQ3xWWZJvSL+7J6slhJJIAg=="""
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
