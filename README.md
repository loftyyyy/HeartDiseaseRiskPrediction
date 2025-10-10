# ❤️ Heart Disease Risk Prediction

A comprehensive machine learning project for predicting heart disease risk using Logistic Regression. This project includes data analysis, model training, and a user-friendly web application deployed on Streamlit Cloud.

## 🌐 Live Demo

**🚀 [Try the Application Now!](https://heart-disease-risk-prediction-ml.streamlit.app/)**

Experience the heart disease risk prediction tool with a clean, intuitive interface that provides instant risk assessments and clinical recommendations.

## 📊 Project Overview

This project analyzes 18 health indicators to predict heart disease risk with **99.1% accuracy**. The model is trained on 70,000 patient records and provides evidence-based recommendations for both high and low-risk patients.

### Key Features
- **Real-time Risk Assessment**: Instant predictions with confidence scores
- **Clinical Recommendations**: Evidence-based suggestions based on risk level
- **User-Friendly Interface**: Clean, responsive design for all devices
- **High Accuracy**: 99.1% accuracy with 99.95% ROC-AUC score
- **Privacy-Focused**: No data storage, all predictions made locally

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.1% |
| **Precision** | 99.1% |
| **Recall** | 99.1% |
| **F1-Score** | 99.1% |
| **ROC-AUC** | 0.9995 |

## 📋 Input Features

The model analyzes 18 comprehensive health indicators:

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

## 🏗️ Project Structure

```
HeartDiseaseRisk/
├── 📁 data/                          # Dataset files
│   ├── heart_disease_risk_dataset_clean.csv
│   └── heart_disease_risk_dataset_earlymed.csv
├── 📁 deployment/                     # Web application
│   ├── app.py                        # Streamlit web app
│   ├── train_model.py                # Model training script
│   ├── heart_disease_model.pkl        # Trained model
│   ├── requirements.txt              # Dependencies
│   └── README.md                     # Deployment guide
├── 📁 scripts/                       # Analysis scripts
│   ├── heart_disease_analysis.py     # Main analysis (THIS IS THE MAIN TRANING)
│   ├── statistical_analysis.py       # Statistical tests
│   ├── data_quality_check.py          # Data validation
│   └── ...                          # Additional analyses
├── 📁 results/                       # Generated outputs
│   ├── model_performance/            # Performance metrics
│   ├── feature_analysis/             # Feature importance
│   ├── visualizations/               # Charts and plots
│   └── statistical/                  # Statistical results
├── 📁 documentation/                 # Project documentation
│   ├── Heart_Disease_Analysis_Explanation.md
│   └── Heart_Disease_Risk_Prediction_Paper.md
└── 📁 analysis/                      # Analysis summaries
    ├── data_quality/                  # Quality reports
    ├── model_performance/             # Performance reports
    └── statistical/                   # Statistical summaries
```

## 🚀 Quick Start

### Option 1: Use the Live Application
Simply visit **[https://heart-disease-risk-prediction-ml.streamlit.app/](https://heart-disease-risk-prediction-ml.streamlit.app/)** to start using the application immediately.

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/loftyyyy/HeartDiseaseRiskPrediction.git
   cd HeartDiseaseRiskPrediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   cd deployment
   python train_model.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Technical Details

- **Algorithm**: Logistic Regression with L1/L2 regularization
- **Training Data**: 70,000 patient records
- **Features**: 18 predictors + 1 target variable
- **Preprocessing**: StandardScaler for feature normalization
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
- **Framework**: Streamlit for web interface
- **Deployment**: Streamlit Cloud

## 📈 Analysis Results

The project includes comprehensive analysis covering:

- **Data Quality Assessment**: Missing values, outliers, and data distribution
- **Statistical Analysis**: Correlation analysis and feature relationships
- **Model Performance**: Detailed evaluation metrics and visualizations
- **Feature Importance**: Understanding which factors contribute most to predictions

## 🛠️ Development

### Training a New Model
```bash
cd deployment
python train_model.py
```

### Running Analysis Scripts
```bash
cd scripts
python heart_disease_analysis.py
```

### Customizing the Web App
- Modify `deployment/app.py` for UI changes
- Update feature descriptions in `get_feature_descriptions()`
- Customize styling in the CSS section

## 📚 Documentation

- **[Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)**: Detailed deployment instructions
- **[Analysis Explanation](documentation/Heart_Disease_Analysis_Explanation.md)**: Comprehensive analysis walkthrough
- **[Research Paper](documentation/Heart_Disease_Risk_Prediction_Paper.md)**: Academic-style project documentation

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

For technical issues or questions about the model, please refer to the documentation or open an issue on GitHub.

## 🏆 Acknowledgments

- Dataset: Heart Disease Risk Dataset (70,000 samples)
- Framework: Streamlit for web deployment
- Machine Learning: Scikit-learn for model training
- Visualization: Matplotlib and Seaborn for analysis

---

**Built with ❤️ for healthcare research and education**

*Last updated: October 2025*
