import cv2
import numpy as np
import tensorflow as tf
try:
    from tensorflow.keras.models import load_model  # type: ignore
except ImportError:
    try:
        from tf_keras.models import load_model  # type: ignore
    except ImportError:
        from tf.keras.models import load_model  # type: ignore
import joblib
from mtcnn import MTCNN
from main import detect_faces, recognize_face
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Verify TensorFlow version
if tf.__version__ != '2.20.0':
    print(f"Warning: Expected TensorFlow 2.20.0, found {tf.__version__}. Some features may not work.")

# Check for opencv-contrib-python
use_kcf = True
try:
    cv2.TrackerKCF_create  # type: ignore
    print("KCF tracking enabled")
except AttributeError:
    use_kcf = False
    print("opencv-contrib-python not found. KCF tracking disabled (webcam FPS ~1-2 instead of ~5-10). "
          "Run in elevated command prompt:\n"
          "1. C:\\Users\\fahmi\\PycharmProjects\\FaceDetRec\\dsenv_tf\\Scripts\\activate\n"
          "2. pip uninstall opencv-python -y\n"
          "3. pip install opencv-contrib-python\n"
          "Then restart the app.")

# Load models
def load_recognition_model():
    try:
        model = load_model('face_recognizer_nn.h5')  # type: ignore
        le = joblib.load('label_encoder.pkl')
        print("Model and label encoder loaded successfully")
        return model, le
    except Exception as e:
        print(f"Failed to load model or label encoder: {e}. Ensure main.py has run successfully and files exist.")
        return None, None

def main():
    model, le = load_recognition_model()
    if model is None or le is None:
        print("Exiting due to model loading failure")
        return

    # Try multiple webcam indices
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam opened successfully on index {index}")
            break
        print(f"Failed to open webcam on index {index}")
        cap.release()
    else:
        print("Error: Could not open webcam on any index (0, 1, 2). Ensure webcam is connected and not in use.")
        return

    frame_count = 0
    tracker = None
    tracker_initialized = False
    last_bbox = None
    identities = []
    use_full_processing = True  # Enable full processing

    if use_kcf:
        try:
            tracker = cv2.TrackerKCF_create()  # type: ignore
            print("KCF tracker initialized")
        except Exception as e:
            print(f"KCF tracker failed: {e}")
            tracker = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.resize(frame, (320, 240))  # Smaller size for faster processing
        frame_count += 1

        # Save debug frame every 100 frames
        if frame_count % 100 == 0:
            debug_path = f"debug_webcam_frame_{frame_count}.jpg"
            cv2.imwrite(debug_path, frame)
            print(f"Saved debug frame: {debug_path}")

        if use_full_processing:
            if use_kcf and tracker is not None and frame_count % 10 != 0 and tracker_initialized:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    face = frame[y:y+h, x:x+w]
                    if face.size > 0 and face.shape[0] >= 32 and face.shape[1] >= 32:
                        identity, confidence = recognize_face(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), model, le)
                        print(f"KCF tracking: {identity}, confidence={confidence:.2f}")
                        identities.append(f"Frame {frame_count}: {identity} (Confidence: {confidence:.2f})")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{identity} ({confidence:.2f})", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                faces, detections = detect_faces(frame)
                if use_kcf and tracker is not None and detections and not tracker_initialized:
                    x, y, w, h = detections[0]['box']
                    tracker.init(frame, (x, y, w, h))
                    tracker_initialized = True
                    last_bbox = (x, y, w, h)
                for i, det in enumerate(detections):
                    if det['confidence'] > 0.6:
                        x, y, w, h = det['box']
                        face = faces[i]
                        if face.shape[0] < 32 or face.shape[1] < 32:
                            print(f"Frame {frame_count}: Skipping small face, shape={face.shape}")
                            continue
                        try:
                            identity, confidence = recognize_face(face, model, le)
                            print(f"Recognized face {i+1}: {identity}, confidence={confidence:.2f}")
                            identities.append(f"Frame {frame_count}: {identity} (Confidence: {confidence:.2f})")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{identity} ({confidence:.2f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            if use_kcf and tracker is not None:
                                tracker = cv2.TrackerKCF_create()  # type: ignore
                                tracker.init(frame, (x, y, w, h))
                                tracker_initialized = True
                                last_bbox = (x, y, w, h)
                        except Exception as e:
                            print(f"Recognition error in frame {frame_count}: {e}")
        else:
            cv2.putText(frame, f"Frame {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}, Identities: {identities[-5:]}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test complete.")
if __name__ == "__main__":
    main()