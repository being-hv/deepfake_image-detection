import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tensorflow as tf

# Must be the first Streamlit command
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
local_css("assets/styles.css")

# --- CORE UTILITIES ---
@st.cache_resource
def load_deepfake_model(model_path):
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return None
    return None

def preprocess_image(image, img_size=(256, 256)):
    image = image.convert('RGB')
    image = image.resize(img_size)
    img_array = np.array(image, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

MODEL_PATH = "models/saved_model/best_model.h5"
model = load_deepfake_model(MODEL_PATH)

if 'history' not in st.session_state:
    st.session_state.history = []

# --- MULTI-PAGE ROUTING ---
st.sidebar.title("Navigation")
menu = ["Home Dashboard", "Upload Detection", "Model Training Dashboard", "Evaluation Metrics", "About Project"]
choice = st.sidebar.radio("Go to", menu)

# Optional sidebar model selection (dummy implementation for UI depth)
st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
model_type = st.sidebar.selectbox("Active Backbone", ["EfficientNetB4", "ResNet50", "Xception"])

if choice == "Home Dashboard":
    st.markdown("<h1 style='text-align: center;'>Deepfake Image Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>Welcome to the Neural Network-powered interface for identifying synthetically manipulated images. Our system leverages advanced transfer learning to separate real human faces from AI-generated deepfakes.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card' style='text-align:center;'><h3>Architecture</h3><p>EfficientNetB4</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card' style='text-align:center;'><h3>Accuracy</h3><p>~96% Target</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card' style='text-align:center;'><h3>Status</h3><p>Online & Active</p></div>", unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", width="stretch", caption="AI Vision Security")

elif choice == "Upload Detection":
    st.title("Real-Time Detection")
    st.markdown("Upload an image or use your webcam to analyze faces.")
    
    tab1, tab2, tab3 = st.tabs(["Drag & Drop Upload", "Webcam Live", "Batch Scanner"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            col_img, col_res = st.columns(2)
            
            with col_img:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_res:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                if st.button("Analyze Image"):
                    if model is None:
                        st.error("No trained model found. Please train first.")
                    else:
                        with st.spinner('Neural Network analyzing pixels...'):
                            processed = preprocess_image(image)
                            pred = model.predict(processed, verbose=0)
                            prob_fake = float(pred[0][0])
                            prob_real = 1.0 - prob_fake
                            
                            label = "FAKE" if prob_fake > 0.5 else "REAL"
                            conf = prob_fake if label == "FAKE" else prob_real
                            
                            color_class = "pred-fake" if label == "FAKE" else "pred-real"
                            st.markdown(f"<div class='pred-text {color_class}'>{label} ({(conf*100):.1f}%)</div>", unsafe_allow_html=True)
                            
                            # Probability chart
                            fig = go.Figure(data=[go.Bar(
                                x=['REAL', 'FAKE / AI'], 
                                y=[prob_real, prob_fake],
                                marker_color=['#00ffcc', '#ff3366']
                            )])
                            fig.update_layout(title="Prediction Probability", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                            st.plotly_chart(fig, width="stretch")
                            
                            # Save to history
                            st.session_state.history.insert(0, {"file": uploaded_file.name, "prediction": label, "confidence": conf})
                st.markdown("</div>", unsafe_allow_html=True)
                
    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            if st.button("Analyze Webcam Frame"):
                if model is None:
                    st.error("No trained model found.")
                else:
                    with st.spinner("Analyzing..."):
                        processed = preprocess_image(image)
                        pred = model.predict(processed, verbose=0)
                        prob_fake = float(pred[0][0])
                        label = "FAKE" if prob_fake > 0.5 else "REAL"
                        conf = prob_fake if label == "FAKE" else (1.0 - prob_fake)
                        st.success(f"Detection: {label} ({conf*100:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        st.markdown("<div class='glass-card'>Scanning a local folder (admin)</div>", unsafe_allow_html=True)
        folder_path = st.text_input("Enter local folder path for batch analysis:")
        if st.button("Scan Folder"):
            st.warning("Not full implemented without backend access on Streamlit cloud.")
                
    st.markdown("### Prediction History")
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, width="stretch")
    else:
        st.info("No predictions yet.")

elif choice == "Model Training Dashboard":
    st.title("Training Dashboard")
    history_file = "models/saved_model/training_history.json"
    
    st.markdown("<div class='glass-card'>Monitor live epoch progress, loss, and accuracy metrics.</div>", unsafe_allow_html=True)
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            hist = json.load(f)
            
        epochs = list(range(1, len(hist['loss']) + 1))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=epochs, y=hist['accuracy'], mode='lines', name='Train Acc', line=dict(color='#00d2ff')))
            if 'val_accuracy' in hist:
                fig_acc.add_trace(go.Scatter(x=epochs, y=hist['val_accuracy'], mode='lines', name='Val Acc', line=dict(color='#ff3366')))
            fig_acc.update_layout(title="Accuracy Curve", paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_acc, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=hist['loss'], mode='lines', name='Train Loss', line=dict(color='#00d2ff')))
            if 'val_loss' in hist:
                fig_loss.add_trace(go.Scatter(x=epochs, y=hist['val_loss'], mode='lines', name='Val Loss', line=dict(color='#ff3366')))
            fig_loss.update_layout(title="Loss Curve", paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_loss, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.warning("No training history found. Run `python src/train.py` first.")
    
    if st.button("Trigger Full Retraining"):
        st.toast("Retraining pipeline triggered (Mock).", icon="⚠️")

elif choice == "Evaluation Metrics":
    st.title("Evaluation Metrics")
    metrics_file = "models/saved_model/metrics.json"
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        col2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        col3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        col4.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
        col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        colA, colB = st.columns(2)
        with colA:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"), 
                                x=['REAL', 'FAKE'], y=['REAL', 'FAKE'],
                                color_continuous_scale='Blues', title="Confusion Matrix")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with colB:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if 'roc_curve' in metrics:
                roc = metrics['roc_curve']
                fig_roc = px.line(x=roc['fpr'], y=roc['tpr'], title="ROC Curve", labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
                fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_roc, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.warning("No evaluation metrics found. Run `python src/evaluate.py` first.")

elif choice == "About Project":
    st.title("About Project")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("""
    ### Deepfake Image Detection
    Building a CNN + transfer learning architecture to fight manipulated digital media.
    
    - **Datasets Used:** 140k Real & Fake Faces, FaceForensics++, DFDC.
    - **Architecture:** EfficientNetB4
    - **Metrics Target:** ~96%
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.subheader("Admin Tools")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=15)
        pdf.cell(200, 10, txt="Deepfake Detection Analytics Report", ln=1, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Generated automatically from Streamlit Dashboard.", ln=2)
        pdf.output("report.pdf")
        
        with open("report.pdf", "rb") as pdf_file:
            st.download_button(label="Download Generated PDF", data=pdf_file, file_name="report.pdf", mime="application/pdf")
            
    st.markdown("---")
    keyword = st.text_input("Dataset Scraper Keyword", value="AI generated person face")
    if st.button("Run Scraper"):
        st.info(f"Triggered scraper for '{keyword}'. Details to follow in backend console.")
