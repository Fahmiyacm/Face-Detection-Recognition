**Face Detection and Recognition System**

This project implements an end-to-end face detection and recognition system using MTCNN for face detection and DeepFace (Facenet) for face recognition. It includes a training pipeline to process the LFW dataset, a webcam-based testing script, and a Streamlit web interface for real-time face recognition. The system uses KCF tracking for improved performance and supports image-based recognition.

**Table of Contents**

**Project Overview**

Features

Requirements

Installation

Project Structure

Usage

Step 1: Prepare the Dataset

Step 2: Train the Model

Step 3: Test with Webcam (Standalone)

Step 4: Run the Streamlit App

Face Detection Issues

"Unknown" Labels

Contributing

License

**Project Overview**

This project processes the LFW dataset to train a face recognition model using Facenet embeddings and a neural network classifier. It uses MTCNN for face detection and supports real-time webcam testing (test_webcam.py) and a Streamlit-based web interface (app.py) for face detection and recognition. The system logs failed images and skipped labels to aid debugging.
Features

Data Preparation: Subsets the LFW dataset, splits it into train/test sets, and applies data augmentation.

Face Detection: Uses MTCNN to detect faces with a configurable confidence threshold (default: 0.5).

Face Recognition: Extracts Facenet embeddings and trains a neural network for identity classification.

Webcam Testing: Real-time face detection and recognition with KCF tracking for improved FPS.

Streamlit Interface: Web app for webcam-based recognition and image upload analysis.

Debugging: Saves debug frames and logs failed images (failed_images.txt) and skipped labels (skipped_labels.txt).

Requirements

Python 3.11

A webcam for real-time testing

LFW dataset (download from kaggle flw)

Dependencies listed in requirements.txt

Installation

Clone the Repository:

git clone https://github.com/<your-username>/FaceDetRec.git

Create and Activate Virtual Environment:
python -m venv dsenv_tf
dsenv_tf\Scripts\activate  # Windows
# source dsenv_tf/bin/activate  # Linux/Mac


Install Dependencies:

pip install -r FaceDetProject/requirements.txt

Download LFW Dataset:

Download the LFW dataset from Kaggle.

Extract it to data/lfw in the project root, ensuring the structure is data/lfw/<person_name>/<image>.jpg.

Verify Installation:

pip list | findstr "tensorflow streamlit streamlit-webrtc av opencv mtcnn deepface torch torchvision pillow scikit-learn numpy joblib matplotlib"

python -c "import av; print(av.__version__)"

python -c "import streamlit; print(streamlit.__version__)"

python -c "import streamlit_webrtc; print(streamlit_webrtc.__version__)"

python -c "import tensorflow; print(tensorflow.__version__)"



Project Structure

FaceDetRec/

├── data/

│   ├── lfw/                    # LFW dataset

│   ├── lfw_subset_800/        # Subset of LFW (generated)

│   ├── train/                 # Training dataset (generated)

│   ├── test/                  # Test dataset (generated)

│   ├── aug_train/             # Augmented training dataset (generated)

├── FaceDetProject/

│   ├── main.py                # Data preparation, training, and testing

│   ├── app.py                 # Streamlit web interface

│   ├── test_webcam.py         # Standalone webcam testing

│   ├── requirements.txt        # Project dependencies

│   ├── face_recognizer_nn.h5  # Trained model (generated)

│   ├── label_encoder.pkl      # Label encoder (generated)

│   ├── accuracy_plot.png      # Training accuracy plot (generated)

│   ├── failed_images.txt      # Failed image logs (generated)

│   ├── skipped_labels.txt     # Skipped test labels (generated)

│   ├── debug_webcam_frame_X.jpg  # Debug frames from webcam (generated)

│   ├── debug_webcam_frame_X_bbox.jpg  # Debug frames with bounding boxes (generated)

├── README.md                  # This file

Usage

Step 1: Prepare the Dataset

Run main.py to prepare the LFW dataset:python FaceDetProject/main.py


This creates data/lfw_subset_800, data/train, data/test, and data/aug_train with augmented images.

Step 2: Train the Model

The same main.py script trains the model and generates:

face_recognizer_nn.h5: Trained neural network model.

label_encoder.pkl: Label encoder for identities.

accuracy_plot.png: Training accuracy plot.

failed_images.txt: Images that failed processing.

skipped_labels.txt: Test labels not in training.


Run:python FaceDetProject/main.py



Step 3: Test with Webcam (Standalone)

Test real-time face detection and recognition:python FaceDetProject/test_webcam.py


Minimal Mode: Edit test_webcam.py and set minimal_mode = True to test webcam feed without processing:minimal_mode = True


Full Mode: Set minimal_mode = False for face detection and recognition.

**Output:**
Window showing webcam feed with frame count (minimal mode, 30 FPS) or bounding boxes and labels (5-10 FPS).
Debug frames saved as debug_webcam_frame_X.jpg and debug_webcam_frame_X_bbox.jpg.
Press q to quit.


Step 4: Run the Streamlit App

Launch the web interface:streamlit run FaceDetProject/app.py


Open http://localhost:8501 in Chrome or Firefox.
Options:
Webcam Stream: Start webcam with "Minimal Mode" (no processing) or "Enable Face Detection and Recognition".
Upload Image: Upload a JPG/PNG for face recognition.
View Accuracy Plot: Display training accuracy plot.



"Unknown" Labels

Problem: Webcam or image recognition only shows "Unknown" labels.
Solution:
Check label_encoder.pkl classes:python -c "import joblib; le=joblib.load('FaceDetProject/label_encoder.pkl'); print(le.classes_)"


If empty or few identities, retrain:python FaceDetProject/main.py


Check failed_images.txt for training issues:type FaceDetProject/failed_images.txt


Check skipped_labels.txt for test/train mismatches:type FaceDetProject/skipped_labels.txt


Ensure data/aug_train and data/test have clear faces and matching identities.

**Contributing**
Contributions are welcome! Please:

Fork the repository.
Create a branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.