import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import joblib
from main import detect_faces, recognize_face
import os

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_recognition_model():
    try:
        model = tf.keras.models.load_model("face_recognizer_nn.h5")
        le = joblib.load("label_encoder.pkl")
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Face Recognition System", layout="centered")
    st.title("🎭 Face Recognition System")
    st.write("A simple and classic interface for **Live Recognition, Image Recognition, and Metrics Plot**")

    menu = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📹 Live Recognition", "🖼️ Image Recognition", "📊 Metrics Plot"]
    )

    # -------------------------------
    # Home Page
    # -------------------------------
    if menu == "🏠 Home":
        st.subheader("Welcome!")
        st.write(
            "This system allows you to perform real-time face recognition, "
            "upload an image for recognition, and view model performance metrics."
        )

    # -------------------------------
    # Live Recognition
    # -------------------------------
    elif menu == "📹 Live Recognition":
        st.subheader("Live Face Recognition")
        st.write("Enable your camera to start recognition")

        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.model, self.le = load_recognition_model()

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                faces, detections = detect_faces(img)
                if self.model is None or self.le is None:
                    return img
                for i, face in enumerate(faces):
                    identity, confidence = recognize_face(face, self.model, self.le)
                    x, y, w, h = detections[i]['box']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, f"{identity} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                return img

        webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
        )

    # -------------------------------
    # Image Recognition
    # -------------------------------
    elif menu == "🖼️ Image Recognition":
        st.subheader("Upload an Image for Recognition")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
        if uploaded_file is not None:
            # Decode uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original_image = image.copy()  # keep a copy of the original
            faces, detections = detect_faces(image)
            model, le = load_recognition_model()
            if model is not None and le is not None and faces:
                for i, face in enumerate(faces):
                    identity, confidence = recognize_face(face, model, le)
                    x, y, w, h = detections[i]['box']
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, f"{identity} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                # Display side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                             caption="Uploaded Image", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                             caption="Recognition Result", use_container_width=True)
                st.success(f"Detected: {identity} (Confidence: {confidence:.2f})")
            else:
                st.warning("No face detected or model not loaded.")

    # -------------------------------
    # Metrics Plot
    # -------------------------------
    elif menu == "📊 Metrics Plot":
        st.subheader("Model Performance Metrics")
        # Display training and validation accuracy plot
        if os.path.exists("accuracy_plot.png"):
            st.image("accuracy_plot.png", caption="Training and Validation Accuracy vs Epochs", use_container_width=True)
        else:
            st.error("No accuracy plot found. Train the model first.")
        # Display test metrics plot (accuracy, precision, recall)
        if os.path.exists("test_metrics_plot.png"):
            st.image("test_metrics_plot.png", caption="Test Accuracy, Precision, and Recall", use_container_width=True)
        else:
            st.error("No test metrics plot found. Run the test metrics computation first.")
        # Display numerical test metrics

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()