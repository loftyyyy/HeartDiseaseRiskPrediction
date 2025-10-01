# Heart Disease Risk Prediction

A Machine Learning project that predicts the risk of heart disease in patients using various health indicators and lifestyle factors.

## 📋 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection and risk assessment can significantly improve patient outcomes and enable preventive care. This project uses machine learning techniques to analyze patient data and predict the likelihood of heart disease.

The model is trained on a comprehensive heart disease dataset containing various medical and demographic features to provide accurate risk predictions.

## 🎯 Objective

The primary goal of this project is to:
- Build a machine learning model that can predict heart disease risk based on patient health data
- Identify the most important factors contributing to heart disease
- Provide a tool for early detection and risk assessment
- Compare different machine learning algorithms to find the best performing model

## 📊 Dataset

The dataset used in this project contains **70,000 records** with **19 features** including various health-related indicators and lifestyle factors:

### Features (Binary: 0 = No, 1 = Yes)
- **Chest_Pain**: Presence of chest pain
- **Shortness_of_Breath**: Difficulty breathing
- **Fatigue**: Excessive tiredness
- **Palpitations**: Irregular heartbeat sensations
- **Dizziness**: Feeling lightheaded or dizzy
- **Swelling**: Body swelling (especially legs/ankles)
- **Pain_Arms_Jaw_Back**: Pain radiating to arms, jaw, or back
- **Cold_Sweats_Nausea**: Cold sweats and nausea
- **High_BP**: High blood pressure
- **High_Cholesterol**: Elevated cholesterol levels
- **Diabetes**: Diabetes diagnosis
- **Smoking**: Smoking habit
- **Obesity**: Obesity condition
- **Sedentary_Lifestyle**: Inactive lifestyle
- **Family_History**: Family history of heart disease
- **Chronic_Stress**: Chronic stress condition
- **Gender**: Gender (0 = Female, 1 = Male)
- **Age**: Age in years (continuous variable)

### Target Variable
- **Heart_Risk**: Heart disease risk (0 = Low Risk, 1 = High Risk)

### Dataset Characteristics
- **Size**: 70,000 samples × 19 features
- **Balance**: Perfectly balanced dataset (35,000 samples per class)
- **Missing Values**: None (complete dataset)
- **Data Types**: All features are numerical (float64)

## 🔧 Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 🚀 Machine Learning Approach

### Data Preprocessing
1. **Data Cleaning**: Handle missing values and outliers
2. **Feature Engineering**: Create new features if needed
3. **Feature Scaling**: Normalize/standardize features for better model performance
4. **Train-Test Split**: Split data into training and testing sets

### Models Evaluated
The project focuses on comparing two key machine learning algorithms:
- **Logistic Regression**: Linear baseline model for binary classification
- **Random Forest**: Ensemble method using multiple decision trees

These models are chosen for their interpretability, performance, and complementary approaches to classification.

### Model Evaluation
Models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to detect disease
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification results

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/loftyyyy/HeartDiseaseRiskPrediction.git
cd HeartDiseaseRiskPrediction
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the following packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## 💻 Usage

### Running the Project

1. **Run the main analysis script**
```bash
cd scripts
python heart_disease_analysis.py
```

2. **For quick analysis (faster)**
```bash
cd scripts
python quick_analysis.py
```

3. **Run individual analysis components**
```bash
cd scripts

# Data quality check
python data_quality_check.py

# Statistical analysis
python statistical_analysis.py

# Duplicate analysis
python duplicate_analysis.py

# Outlier analysis
python iqr_outlier_analysis.py

# Generate histograms
python dataset_distribution_histograms.py
```

4. **View results**
   - Check the `results/` folder for generated visualizations
   - Check the `analysis/` folder for analysis summaries
   - Read the research paper in `documentation/Heart_Disease_Risk_Prediction_Paper.md`

### Making Predictions

```python
# Example code for making predictions
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

# Example patient data
patient_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Make prediction
prediction = model.predict(patient_data)
print(f"Heart Disease Risk: {'High' if prediction[0] == 1 else 'Low'}")
```

## 📈 Results

The models are trained and evaluated on the heart disease dataset. Key findings:

- **Best Performing Model**: [Model name will be determined after training]
- **Accuracy**: [To be updated with actual results]
- **Precision**: [To be updated with actual results]
- **Recall**: [To be updated with actual results]
- **F1-Score**: [To be updated with actual results]

### Important Features
Feature importance analysis reveals the most significant predictors of heart disease:
1. [Feature 1]
2. [Feature 2]
3. [Feature 3]

*Results will be updated after model training and evaluation*

## 🔍 Project Structure

```
HeartDiseaseRiskPrediction/
│
├── data/                      # Dataset files
│   ├── heart_disease_risk_dataset_earlymed.csv  # Original dataset
│   └── heart_disease_risk_dataset_clean.csv     # Cleaned dataset (no duplicates)
│
├── scripts/                   # Python analysis scripts
│   ├── heart_disease_analysis.py        # Main ML pipeline
│   ├── quick_analysis.py                # Fast analysis version
│   ├── data_quality_check.py           # Data quality assessment
│   ├── statistical_analysis.py         # Statistical analysis
│   ├── duplicate_analysis.py           # Duplicate analysis
│   ├── iqr_outlier_analysis.py         # Outlier detection
│   └── dataset_distribution_histograms.py  # Distribution visualization
│
├── analysis/                  # Analysis results and summaries
│   ├── data_quality/          # Data quality reports
│   │   └── iqr_outlier_analysis_summary.csv
│   ├── statistical/           # Statistical summaries
│   │   └── statistical_summary.csv
│   ├── visualizations/        # Visualization data
│   │   └── dataset_distribution_statistics.csv
│   └── model_performance/     # Model performance metrics
│
├── results/                   # Generated visualizations
│   ├── data_quality/         # Data quality plots
│   │   ├── missing_values_heatmap.png
│   │   ├── iqr_outlier_analysis.png
│   │   └── iqr_boxplots.png
│   ├── statistical/          # Statistical visualizations
│   │   ├── age_distribution.png
│   │   ├── data_type_distribution.png
│   │   ├── Figure1_Dataset_Distribution_Histograms.png
│   │   └── statistical_comparison.png
│   ├── model_performance/    # Model evaluation plots
│   │   ├── confusion_matrices.png
│   │   ├── model_comparison.png
│   │   └── roc_curves.png
│   └── feature_analysis/     # Feature analysis plots
│       ├── feature_importance.png
│       ├── feature_importance_comparison.png
│       ├── correlation_matrix.png
│       └── binary_feature_analysis.png
│
├── documentation/             # Project documentation
│   ├── README.md             # Project overview and setup
│   ├── Heart_Disease_Risk_Prediction_Paper.md  # Research paper
│   └── Heart_Disease_Analysis_Explanation.md  # Analysis explanation
│
└── requirements.txt           # Project dependencies
```

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning workflow
- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Model evaluation and comparison
- Healthcare data analysis
- Practical application of classification algorithms

## 🔮 Future Improvements

- [ ] Implement deep learning models (CNN, RNN)
- [ ] Add feature selection techniques
- [ ] Implement cross-validation for robust evaluation
- [ ] Create a web application for easy predictions
- [ ] Add explainability features (SHAP, LIME)
- [ ] Incorporate additional datasets for better generalization
- [ ] Deploy model as an API service
- [ ] Add real-time prediction capabilities

## 📚 References

- UCI Machine Learning Repository - Heart Disease Dataset
- American Heart Association - Heart Disease Statistics
- Research papers on machine learning in healthcare
- Scikit-learn documentation

## 👨‍💻 Author

**loftyyyy**
- GitHub: [@loftyyyy](https://github.com/loftyyyy)

## 📝 License

This project is created as a Machine Learning Final Project for educational purposes.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The open-source community for amazing tools and libraries
- Contributors and supporters of this project

## ⚠️ Disclaimer

This project is for educational and research purposes only. The predictions made by this model should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

---

**Note**: This is a Machine Learning Final Project demonstrating the application of machine learning algorithms to predict heart disease risk based on patient health data.