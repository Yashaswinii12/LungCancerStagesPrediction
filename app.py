# Final Streamlit App with User Registration, Login, Prediction, and Report Retrieval

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
from PIL import Image
from fpdf import FPDF

# Set Streamlit Page Config
st.set_page_config(page_title="Lung Cancer Stage Prediction", page_icon="🫁", layout="wide")


# --- THEME ---
st.markdown(
    """
    <style>
    body { background-color: #000000; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #222222; }
    h1, h2, h3, h4 { color: #00FFFF; } 
    .stButton>button { background-color: #00FFFF; color: black; font-size: 18px; }
    .stTextInput>div>div>input { background-color: #333333; color: white; }
    .stFileUploader>div>div>div>button { background-color: #00FFFF; color: black; }
    </style>
    """, unsafe_allow_html=True
)

# Database connection
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, email TEXT UNIQUE, password TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
    user_id INTEGER, name TEXT, age TEXT, gender TEXT, stage TEXT, confidence REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)''')
conn.commit()

# Auth functions
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def add_user(name, email, password):
    hashed = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email, password):
    cursor.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
    data = cursor.fetchone()
    if data and verify_password(password, data[2]):
        return {"id": data[0], "name": data[1]}
    return None

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_cancer_model (2).h5")

model = load_model()
CLASS_NAMES = ["Benign", "Normal", "Stage 1", "Stage 2", "Stage 3"]
response_dict = {
    "Benign": "😊 Your lungs seem healthy! Regular checkups are always good.",
    "Normal": "✅ No issues detected! Keep up a healthy lifestyle.",
    "Stage 1": "⚠️ A small concern detected. Please consult a doctor.",
    "Stage 2": "🚨 Medium risk detected. A professional checkup is needed soon.",
    "Stage 3": "🚨🚨 High-risk stage! Urgent medical attention is advised."
}
stage_advice = {
    "Benign": "No cancer detected. Maintain a healthy lifestyle and have regular check-ups.",
    "Normal": "Lungs appear healthy. Continue with regular medical check-ups.",
    "Stage 1": "Early-stage cancer detected. Immediate consultation with an oncologist is recommended.",
    "Stage 2": "Moderate-stage cancer detected. Treatment options such as surgery or radiation may be considered.",
    "Stage 3": "Advanced-stage cancer detected. Seek urgent medical attention for further evaluation and treatment."
}

def predict_stage(image):
    img_array = cv2.resize(np.array(image), (224, 224)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    with st.spinner("🔄 Processing image..."):
        prediction = model.predict(img_array)
    stage = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return stage, confidence, prediction

# Chart
def plot_confidence_chart(predictions):
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, predictions[0] * 100, color=['blue', 'green', 'red', 'orange', 'purple'])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence Levels")
    st.pyplot(fig)
def get_img_array(img):
    img = cv2.resize(np.array(img), (224, 224)) / 255.0
    return np.expand_dims(img, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block14_sepconv2_act", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



# Define stage-specific advice
stage_advice = {
    "Stage 0": "No immediate concern. Continue regular checkups and maintain a healthy lifestyle.",
    "Stage I": "Early stage detected. Consult a doctor for further evaluation and monitoring.",
    "Stage II": "Moderate cancer stage. Medical consultation and possible treatment planning advised.",
    "Stage III": "Advanced stage detected. Seek immediate oncology support and begin treatment planning.",
    "Stage IV": "Severe stage detected. Immediate and intensive medical intervention required."
}

def generate_pdf_report():
    report_filename = "Lung_Cancer_Report.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Header with hospital logo and title
    if os.path.exists("hospital_logo.png"):
        pdf.image("hospital_logo.png", x=10, y=8, w=25)

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "MedAI Diagnostics", ln=True, align="C")
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Lung Cancer Stage Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Patient Information
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 10, "Patient Name:", ln=0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, st.session_state.name, ln=1)

    pdf.set_font("Arial", '', 12)
    pdf.cell(40, 10, "Age:", ln=0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, str(st.session_state.age), ln=1)

    pdf.set_font("Arial", '', 12)
    pdf.cell(40, 10, "Gender:", ln=0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, st.session_state.gender, ln=1)
    pdf.ln(5)

    # Prediction Section
    pdf.set_fill_color(240, 248, 255)
    pdf.set_draw_color(100, 100, 100)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Prediction Summary", ln=1, fill=True, border=1)

    pdf.set_font("Arial", '', 12)
    pdf.cell(60, 10, "Predicted Stage:", border=1)
    pdf.cell(0, 10, st.session_state.stage, border=1, ln=1)

    pdf.cell(60, 10, "Confidence Score:", border=1)
    pdf.cell(0, 10, f"{st.session_state.confidence:.2f}%", border=1, ln=1)
    pdf.ln(5)

    # Medical Advice
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(0, 10, "Next Steps:", ln=1)

    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    advice = stage_advice.get(st.session_state.stage, "Consult a doctor for detailed guidance.")
    pdf.multi_cell(0, 10, advice)
    pdf.ln(5)

    # Grad-CAM Image
    if os.path.exists("gradcam_output.jpg"):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 10, "AI Model Focus (Grad-CAM):", ln=1)
        pdf.image("gradcam_output.jpg", w=130)
        pdf.ln(5)

    
    pdf.output(report_filename)
    return report_filename


# SESSION
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = ""
    st.session_state.stage = ""
    st.session_state.confidence = 0.0
    st.session_state.name = ""
    st.session_state.age = ""
    st.session_state.gender = ""

# SIDEBAR
if not st.session_state.logged_in:
    nav_choice = st.sidebar.selectbox("Navigate", ["Login", "Register"])
else:
    nav_choice = st.sidebar.selectbox("Navigate", ["Introduction", "About Dataset", "Prediction", "Download Report",  "Logout"])

# NAVIGATION LOGIC
if nav_choice == "Register":
    st.title(" Register")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if add_user(name, email, password):
            st.success("Registered! Please login.")
        else:
            st.error("User already exists.")

elif nav_choice == "Login":
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(email, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user["id"]
            st.session_state.username = user["name"]
            st.success(f"Welcome {user['name']}!")
            st.rerun()

        else:
            st.error("Invalid login.")

elif nav_choice == "Logout":
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = ""
    st.success("Logged out.")
    st.rerun()

# MAIN PAGES
elif nav_choice == "Introduction":
    st.title("🫁 Lung Cancer Stage Prediction")
    st.image("sample_lung_image.png", use_column_width=True)
    st.write("Upload lung X-ray images to get predictions.")

elif nav_choice == "About Dataset":
    st.title("📊 Dataset Info")
    st.markdown("""Dataset includes:
    - Benign
    - Normal
    - Stage 1, 2, 3""")
    st.image("dataset_sample.jpg", use_column_width=True)

elif nav_choice == "Prediction":
    st.title("🩺 Upload X-ray")
    st.session_state.name = st.session_state.username
    st.write(f"Name: {st.session_state.name}")

    st.session_state.age = st.text_input("Age")
    st.session_state.gender = st.radio("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader("Original X-ray", type=["jpg", "png"])
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original", width=300)
        if st.button("Predict"):
            stage, confidence, predictions = predict_stage(original_image)
            st.session_state.stage = stage
            st.session_state.confidence = confidence
            st.success(f"Prediction: {stage} ({confidence:.2f}%)")
            plot_confidence_chart(predictions)
            st.markdown(f"**AI Says:** {response_dict[stage]}")
            if st.session_state.logged_in:
                cursor.execute("INSERT INTO reports (user_id, name, age, gender, stage, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                               (st.session_state.user_id, st.session_state.name, st.session_state.age, 
                                st.session_state.gender, stage, confidence))
                conn.commit()
            img_array = get_img_array(original_image)
            heatmap = make_gradcam_heatmap(img_array, model)
            # Superimpose on original image
            img = cv2.resize(np.array(original_image), (224, 224))
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            st.image(superimposed_img, caption="Grad-CAM (Model Focus)", use_column_width=True)


elif nav_choice == "Download Report":
    st.title("📄 Report")
    if not st.session_state.stage:
        st.warning("Go to prediction tab first.")
    else:
        st.write(f"Patient: {st.session_state.name}")
        st.write(f"Stage: {st.session_state.stage}")
        st.write(f"Confidence: {st.session_state.confidence:.2f}%")
        if st.button("Generate PDF"):
            pdf = generate_pdf_report()
            with open(pdf, "rb") as f:
                st.download_button("Download PDF", f, file_name="Lung_Cancer_Report.pdf")

