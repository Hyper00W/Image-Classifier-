import cv2
import numpy as np
import pandas as pd
import streamlit as stl
from tensorflow.keras.applications.mobilenet_v2 import(
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_img(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        stl.error(f"Error classifying Image:  {str(e)}")
        return None


def main():
    stl.set_page_config(page_title="AI Image Classifier", page_icon="🥵", layout="centered")
    stl.title("Upload your image and let the AI make its move!")

    @stl.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = stl.file_uploader("Choose an image. . .", type=["jpg","png"])

    if uploaded_file is not None:
        image = stl.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = stl.button("Classify Image")

        if btn:
            with stl.spinner("Analyzing Image...."):
                image = Image.open(uploaded_file)
                predictions = classify_img(model, image)

                if predictions:
                    stl.subheader("Predictions")
                    for _, label , score in predictions:
                        stl.write(f"**{label}**:{score:.2%}")

if __name__ == "__main__":
    main()