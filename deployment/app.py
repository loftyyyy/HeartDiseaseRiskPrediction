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
def load_models():
    """Load the trained models and metadata."""
    try:
        with open('deployment/heart_disease_models.pkl', 'rb') as f:
            models_data = pickle.load(f)
        return models_data
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

def create_model_selection():
    """Create model selection interface."""
    st.subheader("🤖 Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📊 Choose Your Model:</h4>
        <p>Select which machine learning algorithm to use for prediction:</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_choice = st.selectbox(
            "Select Model",
            options=["logistic_regression", "random_forest"],
            format_func=lambda x: "Logistic Regression" if x == "logistic_regression" else "Random Forest",
            help="Choose between Logistic Regression and Random Forest algorithms"
        )
    
    return model_choice

def create_input_mode_selection():
    """Create input mode selection (single vs batch)."""
    st.subheader("📝 Input Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📋 Choose Input Method:</h4>
        <p>Select how you want to input patient data:</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        input_mode = st.radio(
            "Input Mode",
            options=["single", "batch"],
            format_func=lambda x: "Single Patient Form" if x == "single" else "CSV Batch Upload",
            help="Choose between single patient form or CSV batch upload"
        )
    
    return input_mode

def create_csv_upload_section():
    """Create CSV upload section."""
    st.subheader("📁 CSV Upload")
    
    # Upload CSV file
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with patient data. The file should contain the same columns as the training data."
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display basic info about the uploaded data
            st.markdown(f"""
            <div class="info-box">
            <h4>📊 Uploaded Data Summary:</h4>
            <p><strong>Records:</strong> {len(df)} patients<br>
            <strong>Columns:</strong> {len(df.columns)} features<br>
            <strong>File Size:</strong> {uploaded_file.size:,} bytes</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show preview of the data
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check if required columns are present
            expected_features = [
                'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
                'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP', 'High_Cholesterol',
                'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle', 'Family_History',
                'Chronic_Stress', 'Gender', 'Age'
            ]
            
            missing_columns = [col for col in expected_features if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.markdown("""
                **Required columns:** Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness,
                Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea, High_BP, High_Cholesterol, Diabetes,
                Smoking, Obesity, Sedentary_Lifestyle, Family_History, Chronic_Stress, Gender, Age
                """)
                return None
            else:
                st.success("✅ All required columns are present!")
                return df
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return None
    
    return None

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

def make_prediction(models_data, selected_model, inputs):
    """Make prediction using the selected model."""
    # Get the expected feature order from the models data
    expected_features = models_data['feature_names']
    
    # Reorder inputs to match the expected feature order
    ordered_inputs = [inputs[feature] for feature in expected_features]
    
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([ordered_inputs], columns=expected_features)
    
    # Get the selected model data
    model_data = models_data['models'][selected_model]
    
    # Scale the features if needed
    if model_data['model_info']['needs_scaling']:
        scaled_input = model_data['scaler'].transform(input_df)
    else:
        scaled_input = input_df
    
    # Make prediction
    prediction = model_data['model'].predict(scaled_input)[0]
    probability = model_data['model'].predict_proba(scaled_input)[0]
    
    return prediction, probability, model_data

def make_batch_predictions(models_data, selected_model, df):
    """Make predictions for multiple records using the selected model."""
    # Get the expected feature order from the models data
    expected_features = models_data['feature_names']
    
    # Ensure the DataFrame has the correct column order
    input_df = df[expected_features].copy()
    
    # Get the selected model data
    model_data = models_data['models'][selected_model]
    
    # Scale the features if needed
    if model_data['model_info']['needs_scaling']:
        scaled_input = model_data['scaler'].transform(input_df)
    else:
        scaled_input = input_df
    
    # Make predictions
    predictions = model_data['model'].predict(scaled_input)
    probabilities = model_data['model'].predict_proba(scaled_input)
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['Prediction'] = predictions
    results_df['Risk_Level'] = results_df['Prediction'].map({0: 'Low Risk', 1: 'High Risk'})
    results_df['Confidence_Low_Risk'] = probabilities[:, 0]
    results_df['Confidence_High_Risk'] = probabilities[:, 1]
    results_df['Model_Used'] = model_data['model_info']['algorithm']
    
    return results_df, model_data

def display_prediction(prediction, probability, model_data):
    """Display the prediction result."""
    st.subheader("🎯 Risk Assessment Result")
    
    # Show which model was used
    algorithm = model_data['model_info']['algorithm']
    st.markdown(f"""
    <div class="info-box">
    <strong>🤖 Model Used:</strong> {algorithm}<br>
    </div>
    """, unsafe_allow_html=True)
    
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

def display_batch_results(results_df, model_data):
    """Display batch prediction results."""
    st.subheader("🎯 Batch Prediction Results")
    
    # Show which model was used
    algorithm = model_data['model_info']['algorithm']
    st.markdown(f"""
    <div class="info-box">
    <strong>🤖 Model Used:</strong> {algorithm}<br>
    <strong>📊 Performance:</strong> {model_data['performance_metrics']['accuracy']:.1%} accuracy<br>
    <strong>📈 Records Processed:</strong> {len(results_df)} patients
    </div>
    """, unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk_count = len(results_df[results_df['Prediction'] == 1])
        st.metric("High Risk Patients", high_risk_count)
    
    with col2:
        low_risk_count = len(results_df[results_df['Prediction'] == 0])
        st.metric("Low Risk Patients", low_risk_count)
    
    with col3:
        avg_confidence = results_df['Confidence_High_Risk'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Display results table
    st.subheader("📊 Detailed Results")
    
    # Select columns to display
    display_columns = ['Age', 'Gender', 'Risk_Level', 'Confidence_High_Risk', 'Confidence_Low_Risk']
    available_columns = [col for col in display_columns if col in results_df.columns]
    
    st.dataframe(
        results_df[available_columns].style.format({
            'Confidence_High_Risk': '{:.1%}',
            'Confidence_Low_Risk': '{:.1%}'
        }),
        use_container_width=True
    )
    
    # Download button
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"heart_disease_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the complete prediction results including all original data and predictions"
    )

def display_model_info(models_data):
    """Display model information."""
    st.sidebar.subheader("📊 Available Models")
    
    # Show both models
    for model_name, model_data in models_data['models'].items():
        algorithm = model_data['model_info']['algorithm']
        metrics = model_data['performance_metrics']
        
        st.sidebar.markdown(f"""
        <div class="model-info-box">
        <strong>🤖 {algorithm}:</strong><br>
        Accuracy: {metrics['accuracy']:.1%}<br>
        Precision: {metrics['precision']:.1%}<br>
        Recall: {metrics['recall']:.1%}<br>
        F1-Score: {metrics['f1_score']:.1%}<br>
        ROC-AUC: {metrics['roc_auc']:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="model-info-box">
    <strong>🔧 Dataset Info:</strong><br>
    Features: {models_data['features']}<br>
    Dataset: {models_data['dataset_size']}<br>
    Trained: {models_data['training_date']}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Prediction</h1>', unsafe_allow_html=True)
    
    # Load models
    models_data = load_models()
    if models_data is None:
        st.stop()
    
    # Display model info in sidebar
    display_model_info(models_data)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
    <h4>⚠️ Important Disclaimer:</h4>
    <p><strong>This tool is for educational and research purposes only.</strong> It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    selected_model = create_model_selection()
    
    # Input mode selection
    input_mode = create_input_mode_selection()
    
    if input_mode == "single":
        # Single patient input form
        inputs = create_input_form()
        
        # Prediction button
        if st.button("🔍 Assess Heart Disease Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                prediction, probability, model_data = make_prediction(models_data, selected_model, inputs)
                display_prediction(prediction, probability, model_data)
    
    else:
        # Batch CSV upload
        uploaded_df = create_csv_upload_section()
        
        if uploaded_df is not None:
            # Batch prediction button
            if st.button("🔍 Analyze All Patients", type="primary", use_container_width=True):
                with st.spinner("Analyzing patient data..."):
                    results_df, model_data = make_batch_predictions(models_data, selected_model, uploaded_df)
                    display_batch_results(results_df, model_data)
    
    # Additional information
    st.markdown("---")
    st.subheader("📚 About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **How it works:**
        - Choose between Logistic Regression and Random Forest
        - Single patient form or CSV batch upload
        - Trained on 70,000 patient records
        - Analyzes 18 health indicators
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - 99.1%+ accuracy on test data
        - Batch processing for multiple patients
        - Download results as CSV
        - Evidence-based recommendations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Heart Disease Risk Prediction System | Machine Learning Research Project</p>
    <p>Built with Streamlit | Powered by Logistic Regression & Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

