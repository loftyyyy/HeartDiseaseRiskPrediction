"""
Heart Disease Risk Prediction - Streamlit Web Application
A user-friendly interface for predicting heart disease risk using Logistic Regression
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved contrast
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333;
    }
    .prediction-high {
        background-color: #ffcdd2;
        color: #b71c1c;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #d32f2f;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.3);
    }
    .prediction-low {
        background-color: #c8e6c9;
        color: #1b5e20;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #388e3c;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(56, 142, 60, 0.3);
    }
    .info-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333;
    }
    .disclaimer-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #e65100;
    }
    .model-info-box {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #9c27b0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #4a148c;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model and metadata."""
    try:
        with open('deployment/heart_disease_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")
        return None

def get_feature_descriptions():
    """Get human-readable descriptions for features."""
    return {
        'Chest_Pain': 'Chest Pain',
        'Shortness_of_Breath': 'Shortness of Breath',
        'Fatigue': 'Fatigue',
        'Palpitations': 'Palpitations',
        'Dizziness': 'Dizziness',
        'Swelling': 'Swelling (Legs/Ankles)',
        'Pain_Arms_Jaw_Back': 'Pain in Arms, Jaw, or Back',
        'Cold_Sweats_Nausea': 'Cold Sweats and Nausea',
        'High_BP': 'High Blood Pressure',
        'High_Cholesterol': 'High Cholesterol',
        'Diabetes': 'Diabetes',
        'Smoking': 'Smoking',
        'Obesity': 'Obesity',
        'Sedentary_Lifestyle': 'Sedentary Lifestyle',
        'Family_History': 'Family History of Heart Disease',
        'Chronic_Stress': 'Chronic Stress',
        'Gender': 'Gender',
        'Age': 'Age'
    }

def create_input_form():
    """Create the input form for user data."""
    st.subheader("📋 Patient Information")
    
    feature_descriptions = get_feature_descriptions()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    inputs = {}
    
    with col1:
        st.write("**Demographics**")
        inputs['Age'] = st.slider(
            "Age (years)", 
            min_value=20, 
            max_value=100, 
            value=50,
            help="Patient's age in years"
        )
        
        inputs['Gender'] = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Patient's gender"
        )
        
        st.write("**Symptoms**")
        inputs['Chest_Pain'] = st.selectbox(
            "Chest Pain",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Presence of chest pain"
        )
        
        inputs['Shortness_of_Breath'] = st.selectbox(
            "Shortness of Breath",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Difficulty breathing"
        )
        
        inputs['Fatigue'] = st.selectbox(
            "Fatigue",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Excessive tiredness"
        )
        
        inputs['Palpitations'] = st.selectbox(
            "Palpitations",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Irregular heartbeat sensations"
        )
        
        inputs['Dizziness'] = st.selectbox(
            "Dizziness",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Feeling lightheaded or dizzy"
        )
        
        inputs['Swelling'] = st.selectbox(
            "Swelling (Legs/Ankles)",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Body swelling, especially in legs/ankles"
        )
        
        inputs['Pain_Arms_Jaw_Back'] = st.selectbox(
            "Pain in Arms, Jaw, or Back",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Pain radiating to arms, jaw, or back"
        )
        
        inputs['Cold_Sweats_Nausea'] = st.selectbox(
            "Cold Sweats and Nausea",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Cold sweats and nausea"
        )
    
    with col2:
        st.write("**Medical History**")
        inputs['High_BP'] = st.selectbox(
            "High Blood Pressure",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="History of high blood pressure"
        )
        
        inputs['High_Cholesterol'] = st.selectbox(
            "High Cholesterol",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Elevated cholesterol levels"
        )
        
        inputs['Diabetes'] = st.selectbox(
            "Diabetes",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Diabetes diagnosis"
        )
        
        st.write("**Lifestyle Factors**")
        inputs['Smoking'] = st.selectbox(
            "Smoking",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Smoking habit"
        )
        
        inputs['Obesity'] = st.selectbox(
            "Obesity",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Obesity condition"
        )
        
        inputs['Sedentary_Lifestyle'] = st.selectbox(
            "Sedentary Lifestyle",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Inactive lifestyle"
        )
        
        inputs['Family_History'] = st.selectbox(
            "Family History of Heart Disease",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Family history of heart disease"
        )
        
        inputs['Chronic_Stress'] = st.selectbox(
            "Chronic Stress",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Chronic stress condition"
        )
    
    return inputs

def make_prediction(model_data, inputs):
    """Make prediction using the trained model."""
    # Get the expected feature order from the model
    expected_features = model_data['feature_names']
    
    # Reorder inputs to match the expected feature order
    ordered_inputs = [inputs[feature] for feature in expected_features]
    
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([ordered_inputs], columns=expected_features)
    
    # Scale the features
    scaled_input = model_data['scaler'].transform(input_df)
    
    # Make prediction
    prediction = model_data['model'].predict(scaled_input)[0]
    probability = model_data['model'].predict_proba(scaled_input)[0]
    
    return prediction, probability

def display_prediction(prediction, probability):
    """Display the prediction result."""
    st.subheader("🎯 Risk Assessment Result")
    
    if prediction == 1:
        st.markdown('<div class="prediction-high">⚠️ HIGH HEART DISEASE RISK</div>', unsafe_allow_html=True)
        st.markdown(f"**Confidence Level: {probability[1]*100:.1f}%**")
        st.markdown(f"*The model is {probability[1]*100:.1f}% confident this patient has HIGH heart disease risk*")
        
        st.markdown("""
        <div class="info-box">
        <h4>📋 Recommendations for High Risk:</h4>
        <ul>
        <li><strong>Consult with a cardiologist immediately</strong></li>
        <li>Consider immediate medical evaluation</li>
        <li>Monitor symptoms closely</li>
        <li>Follow up with healthcare provider</li>
        <li>Consider lifestyle modifications</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-low">✅ LOW HEART DISEASE RISK</div>', unsafe_allow_html=True)
        st.markdown(f"**Confidence Level: {probability[0]*100:.1f}%**")
        st.markdown(f"*The model is {probability[0]*100:.1f}% confident this patient has LOW heart disease risk*")
        
        st.markdown("""
        <div class="info-box">
        <h4>📋 Recommendations for Low Risk:</h4>
        <ul>
        <li><strong>Continue regular health checkups</strong></li>
        <li>Maintain healthy lifestyle</li>
        <li>Monitor any new symptoms</li>
        <li>Follow preventive care guidelines</li>
        <li>Stay active and eat well</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def display_model_info(model_data):
    """Display model information."""
    st.sidebar.subheader("📊 Model Information")
    
    metrics = model_data['performance_metrics']
    st.sidebar.markdown(f"""
    <div class="model-info-box">
    <strong>🎯 Model Performance:</strong><br>
    Accuracy: {metrics['accuracy']:.1%}<br>
    Precision: {metrics['precision']:.1%}<br>
    Recall: {metrics['recall']:.1%}<br>
    F1-Score: {metrics['f1_score']:.1%}<br>
    ROC-AUC: {metrics['roc_auc']:.3f}
    </div>
    """, unsafe_allow_html=True)
    
    model_info = model_data['model_info']
    st.sidebar.markdown(f"""
    <div class="model-info-box">
    <strong>🔧 Model Details:</strong><br>
    Algorithm: {model_info['algorithm']}<br>
    Features: {model_info['features']}<br>
    Dataset: {model_info['dataset_size']}<br>
    Trained: {model_info['training_date']}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    # Display model info in sidebar
    display_model_info(model_data)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
    <h4>⚠️ Important Disclaimer:</h4>
    <p><strong>This tool is for educational and research purposes only.</strong> It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    inputs = create_input_form()
    
    # Prediction button
    if st.button("🔍 Assess Heart Disease Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            prediction, probability = make_prediction(model_data, inputs)
            display_prediction(prediction, probability)
    
    # Additional information
    st.markdown("---")
    st.subheader("📚 About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **How it works:**
        - Uses Logistic Regression machine learning model
        - Trained on 70,000 patient records
        - Analyzes 18 health indicators
        - Provides risk probability assessment
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - 99.1% accuracy on test data
        - Fast prediction (< 1 second)
        - User-friendly interface
        - Evidence-based recommendations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Heart Disease Risk Prediction System | Machine Learning Research Project</p>
    <p>Built with Streamlit | Powered by Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

