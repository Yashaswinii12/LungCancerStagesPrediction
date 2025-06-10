Lung Cancer Stage Prediction using Hybrid CNN (Xception + MobileNetV2)
This project uses deep learning to classify lung cancer stages (Normal, Benign, Stage 1, Stage 2, Stage 3) from X-ray images. A hybrid CNN model combining Xception and MobileNetV2 is trained on preprocessed image data to achieve high accuracy. The application is deployed using Streamlit for an interactive and user-friendly interface.

 Tech Stack
Language: Python
Deep Learning: TensorFlow / Keras
Pretrained Models: Xception, MobileNetV2
Frontend: Streamlit
Visualization: Matplotlib, Seaborn
Image Processing: OpenCV
File Handling: OS, Zipfile, Shutil
Evaluation: Scikit-learn (Confusion Matrix, Classification Report)

Model Architecture
The model combines features from two powerful CNN architectures:
Xception: Efficient at extracting fine-grained spatial features.
MobileNetV2: Lightweight, optimized for mobile and embedded systems.
The extracted features are concatenated and passed through dense layers with BatchNormalization, Dropout, and GlobalAveragePooling2D.

Data Input
Input: Chest X-ray images (.jpg, .png)
Output: Predicted Stage — Normal, Benign, Stage 1, Stage 2, Stage 3

Evaluation Metrics
Accuracy
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix

Grad-CAM (for visualizing model attention)

Streamlit Web App
Features:
Upload chest X-ray image

Predict cancer stage with confidence score

View Grad-CAM visual explanation

Download PDF report

Run the app:
streamlit run streamlit_app.py
