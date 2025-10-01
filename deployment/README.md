# Heart Disease Risk Prediction - Streamlit Deployment

A user-friendly web application for predicting heart disease risk using Logistic Regression machine learning model.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Features

- **User-Friendly Interface**: Clean, intuitive design with organized input forms
- **Real-Time Prediction**: Instant risk assessment with probability scores
- **Clinical Recommendations**: Evidence-based suggestions based on risk level
- **Model Information**: Display of model performance metrics and details
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 Model Performance

- **Accuracy**: 99.1%
- **Precision**: 99.1%
- **Recall**: 99.1%
- **F1-Score**: 99.1%
- **ROC-AUC**: 0.9995

## 📊 Input Features

The application analyzes 18 health indicators:

### Demographics
- Age
- Gender

### Symptoms
- Chest Pain
- Shortness of Breath
- Fatigue
- Palpitations
- Dizziness
- Swelling (Legs/Ankles)
- Pain in Arms, Jaw, or Back
- Cold Sweats and Nausea

### Medical History
- High Blood Pressure
- High Cholesterol
- Diabetes

### Lifestyle Factors
- Smoking
- Obesity
- Sedentary Lifestyle
- Family History of Heart Disease
- Chronic Stress

## 🔧 Technical Details

- **Algorithm**: Logistic Regression with L1/L2 regularization
- **Training Data**: 70,000 patient records
- **Features**: 18 predictors + 1 target variable
- **Preprocessing**: StandardScaler for feature normalization
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation

## 📁 File Structure

```
deployment/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── heart_disease_model.pkl # Trained model (generated)
└── feature_names.txt     # Feature list (generated)
```

## ⚠️ Important Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 🛠️ Development

### Training a New Model
1. Ensure the dataset is available at `../data/heart_disease_risk_dataset_earlymed.csv`
2. Run `python train_model.py` to train and save the model
3. The model will be saved as `heart_disease_model.pkl`

### Customizing the Interface
- Modify `app.py` to change the UI layout
- Update feature descriptions in `get_feature_descriptions()`
- Customize styling in the CSS section

### Adding New Features
- Update the input form in `create_input_form()`
- Modify the model training script if needed
- Update feature descriptions and help text

## 📈 Performance Optimization

- Model is cached using `@st.cache_data` for faster loading
- Predictions are made in real-time (< 1 second)
- Optimized for both accuracy and speed
- Memory efficient with small model size

## 🔒 Security Considerations

- No patient data is stored or logged
- All predictions are made locally
- No external API calls
- Privacy-focused design

## 📞 Support

For technical issues or questions about the model, please refer to the main project documentation or contact the development team.

---

**Built with ❤️ using Streamlit and Scikit-learn**

