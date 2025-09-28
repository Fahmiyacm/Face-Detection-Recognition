import os
import random
import shutil
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image
from mtcnn import MTCNN
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
try:
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, Callback
except ImportError:
    try:
        from tf_keras.models import Sequential, load_model  # type: ignore
        from tf_keras.layers import Dense, Dropout
        from tf_keras.optimizers import Adam
        from tf_keras.callbacks import EarlyStopping, Callback
    except ImportError:
        from tf.keras.models import Sequential, load_model  # type: ignore
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Verify TensorFlow version
if tf.__version__ != '2.20.0':
    print(f"Warning: Expected TensorFlow 2.20.0, found {tf.__version__}. Some features may not work.")

# Custom callback for precision and recall during validation
class MetricsCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_data
        y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        logs['val_precision'] = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        logs['val_recall'] = recall_score(y_val, y_pred, average='weighted', zero_division=0)

# Phase 1: Data Collection and Preparation
def prepare_data(num_identities=100):
    lfw_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\lfw"
    subset_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\lfw_subset_800"
    train_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\train"
    test_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\test"
    aug_train_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\aug_train"

    if os.path.exists(aug_train_path) and os.listdir(aug_train_path):
        print("Using existing aug_train dataset. Skipping data preparation.")
        return

    os.makedirs(subset_path, exist_ok=True)
    person_folders = [f for f in os.listdir(lfw_path) if os.path.isdir(os.path.join(lfw_path, f))]
    selected_persons = random.sample(person_folders, min(num_identities, len(person_folders)))

    for person in selected_persons:
        src = os.path.join(lfw_path, person)
        dst = os.path.join(subset_path, person)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"Subset created with {len(selected_persons)} individuals.")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for person in os.listdir(subset_path):
        person_dir = os.path.join(subset_path, person)
        if os.path.isdir(person_dir):
            images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
            if len(images) == 0:
                continue
            elif len(images) == 1:
                train_person_dir = os.path.join(train_path, person)
                os.makedirs(train_person_dir, exist_ok=True)
                shutil.copy(os.path.join(person_dir, images[0]), os.path.join(train_person_dir, images[0]))
            else:
                train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
                train_person_dir = os.path.join(train_path, person)
                test_person_dir = os.path.join(test_path, person)
                os.makedirs(train_person_dir, exist_ok=True)
                os.makedirs(test_person_dir, exist_ok=True)
                for img in train_imgs:
                    shutil.copy(os.path.join(person_dir, img), os.path.join(train_person_dir, img))
                for img in test_imgs:
                    shutil.copy(os.path.join(person_dir, img), os.path.join(test_person_dir, img))

    print("Dataset split complete.")

    augment = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
    ])

    os.makedirs(aug_train_path, exist_ok=True)

    for person in os.listdir(train_path):
        person_dir = os.path.join(train_path, person)
        aug_person_dir = os.path.join(aug_train_path, person)
        os.makedirs(aug_person_dir, exist_ok=True)
        for img_name in os.listdir(person_dir):
            if img_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(person_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                shutil.copy(img_path, os.path.join(aug_person_dir, img_name))
                aug_image = augment(image)
                aug_image = transforms.ToPILImage()(aug_image)
                aug_name = f"aug_{img_name}"
                aug_image.save(os.path.join(aug_person_dir, aug_name))

    print("Augmentation complete. Use aug_train for training.")

# Phase 2: Face Detection
def detect_faces(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        detections = detector.detect_faces(image_rgb)
        faces = []
        for det in detections:
            if det['confidence'] > 0.7:  # Lowered threshold
                x, y, w, h = det['box']
                x, y = max(0, x), max(0, y)
                face = image_rgb[y:y+h, x:x+w]
                if face.size > 0 and face.shape[0] >= 32 and face.shape[1] >= 32:
                    faces.append(face)
                print(f"Detected face: confidence={det['confidence']:.2f}, box=({x},{y},{w},{h})")
            else:
                print(f"Low confidence detection: {det['confidence']:.2f}")
        if not faces:
            print("No faces detected in image")
        return faces, detections
    except Exception as e:
        print(f"Face detection error: {e}")
        return [], []

# Phase 3: Face Recognition
def train_model(num_identities=100):
    train_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\aug_train"
    model_path = 'face_recognizer_nn.h5'
    le_path = 'label_encoder.pkl'

    if os.path.exists(model_path) and os.path.exists(le_path):
        print("Loading existing model and label encoder.")
        model = load_model(model_path)  # type: ignore
        le = joblib.load(le_path)
        return model, le

    X_train = []
    y_train = []
    failed_images = []
    total_images = sum(len(os.listdir(os.path.join(train_path, person))) for person in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, person)))
    processed = 0

    for person in os.listdir(train_path):
        person_dir = os.path.join(train_path, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        failed_images.append(img_path)
                        continue
                    faces, _ = detect_faces(img)
                    processed += 1
                    if faces:
                        try:
                            embedding = DeepFace.represent(faces[0], model_name='Facenet', detector_backend='skip')
                            X_train.append(embedding[0]['embedding'])
                            y_train.append(person)
                            print(f"Processed {processed}/{total_images} images: {img_path}")
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            failed_images.append(img_path)
                    else:
                        print(f"No valid faces detected in {img_path}")
                        failed_images.append(img_path)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    joblib.dump(le, le_path)
    print(f"Collected {len(X_train)} embeddings for {len(np.unique(y_train))} identities.")

    X_train = np.array(X_train)
    if len(X_train) == 0:
        print("X_train is empty. Check dataset.")
        return None, None

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(128,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(y_train_encoded)), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    metrics_callback = MetricsCallback((X_val, y_val))

    history = model.fit(X_tr, y_tr, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, metrics_callback])

    model.save(model_path)

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    print("Training complete. Accuracy plot saved.")

    with open('failed_images.txt', 'a') as f:
        f.write('\n'.join(failed_images))

    return model, le

# Test existing model
def test_model(model, le):
    test_path = r"C:\Users\fahmi\PycharmProjects\FaceDetRec\data\test"
    failed_images = []
    X_test = []
    y_test = []
    total_test_images = sum(len(os.listdir(os.path.join(test_path, person))) for person in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, person)))
    processed = 0
    skipped_labels = []

    for person in os.listdir(test_path):
        person_dir = os.path.join(test_path, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load test image: {img_path}")
                        failed_images.append(img_path)
                        continue
                    faces, _ = detect_faces(img)
                    processed += 1
                    if faces:
                        try:
                            embedding = DeepFace.represent(faces[0], model_name='Facenet', detector_backend='skip')
                            X_test.append(embedding[0]['embedding'])
                            y_test.append(person)
                            print(f"Processed {processed}/{total_test_images} test images: {img_path}")
                        except Exception as e:
                            print(f"Error processing test {img_path}: {e}")
                            failed_images.append(img_path)
                    else:
                        print(f"No valid faces detected in test {img_path}")
                        failed_images.append(img_path)

    valid_labels = [label for label in y_test if label in le.classes_]
    valid_indices = [i for i, label in enumerate(y_test) if label in le.classes_]
    skipped_labels = list(set(y_test) - set(le.classes_))

    if skipped_labels:
        print(f"Skipped test labels not in training: {skipped_labels}")
        with open('skipped_labels.txt', 'w') as f:
            f.write('\n'.join(skipped_labels))

    X_test = [X_test[i] for i in valid_indices]
    y_test = valid_labels

    if len(X_test) == 0:
        print("No valid test labels after filtering. Check dataset consistency.")
        return

    y_test_encoded = le.transform(y_test)
    X_test = np.array(X_test)

    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)

    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}")

    # Save bar plot for test metrics
    metrics = ['Accuracy', 'Precision', 'Recall']
    values = [accuracy, precision, recall]
    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Test Metrics')
    plt.savefig('test_metrics_plot.png')
    plt.close()

    print("Test metrics plot saved as 'test_metrics_plot.png'")

    with open('failed_images.txt', 'a') as f:
        f.write('\n'.join(failed_images))

# Recognize face function for inference
def recognize_face(face_image, model, le):
    try:
        if face_image.shape[0] < 32 or face_image.shape[1] < 32:
            print("Face image too small for recognition")
            return "Unknown", 0.0
        embedding = DeepFace.represent(face_image, model_name='Facenet', detector_backend='skip')[0]['embedding']
        pred = model.predict(np.array([embedding]), verbose=0)
        label_idx = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)
        if confidence > 0.7:
            label = le.inverse_transform([label_idx])[0]
            print(f"Recognized: {label}, confidence={confidence:.2f}. Known labels: {list(le.classes_)}")
            return label, confidence
        print(f"Low confidence: {confidence:.2f}. Labeling as Unknown. Known labels: {list(le.classes_)}")
        return "Unknown", confidence
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown", 0.0