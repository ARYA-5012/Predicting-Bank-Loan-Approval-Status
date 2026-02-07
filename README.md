# 🏦 Bank Loan Approval Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.4+-orange.svg)](https://spark.apache.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Big Data Machine Learning pipeline** built with PySpark to predict bank loan approval status. This project implements and compares 5 classification algorithms to help financial institutions streamline their loan approval process and assess credit risk effectively.

---

## 📋 Problem Statement

Banks receive thousands of loan applications daily. Manual processing is:
- ⏱️ Time-consuming
- 😓 Prone to human error
- 📊 Inconsistent across reviewers

This project builds a **classification model** that predicts whether a loan application will be **Approved (Y)** or **Rejected (N)** based on applicant information.

---

## 🎯 Features

- **5 ML Models Compared**: Logistic Regression, Decision Tree, Random Forest, SVM, and Gradient Boosted Trees
- **Automated Preprocessing**: Handles missing values, encodes categorical features, scales numeric features
- **Feature Engineering**: Creates derived features like Combined Income and Income-to-Loan Ratio
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics
- **Rich Visualizations**: EDA plots, confusion matrices, ROC curves, and model comparison charts

---

## 📊 Dataset

The project uses a loan approval dataset with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| Gender | Male/Female | Categorical |
| Married | Yes/No | Categorical |
| Dependents | Number of dependents | Categorical |
| Education | Graduate/Not Graduate | Categorical |
| Self_Employed | Yes/No | Categorical |
| ApplicantIncome | Applicant's income | Numeric |
| CoapplicantIncome | Co-applicant's income | Numeric |
| LoanAmount | Loan amount (in thousands) | Numeric |
| Loan_Amount_Term | Term of loan (in months) | Numeric |
| Credit_History | Credit history meets guidelines (1/0) | Numeric |
| Property_Area | Urban/Semiurban/Rural | Categorical |
| **Loan_Status** | Loan approved (Y/N) - **Target** | Categorical |

### Getting the Dataset

1. Download from [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
2. Rename the file to `loan_data.csv`
3. Place it in the project root directory

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Java 8 or 11 (required for PySpark)
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Predicting-Bank-Loan-Approval-Status.git
cd Predicting-Bank-Loan-Approval-Status

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
python Bank_Loan_Pro.py
```

---

## 📁 Project Structure

```
Predicting-Bank-Loan-Approval-Status/
│
├── Bank_Loan_Pro.py           # Main ML pipeline script
├── requirements.txt           # Python dependencies
├── loan_data.csv              # Dataset (download separately)
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
│
└── outputs/                   # Generated visualizations
    ├── loan_status_distribution.png
    ├── income_by_status.png
    ├── credit_history_approval.png
    ├── confusion_matrix_*.png
    ├── roc_curves_comparison.png
    ├── model_comparison.png
    └── feature_importance_*.png
```

---

## 🤖 Models Implemented

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model for binary classification |
| **Decision Tree** | Tree-based model with interpretable rules |
| **Random Forest** | Ensemble of decision trees |
| **SVM (LinearSVC)** | Support Vector Machine with linear kernel |
| **Gradient Boosted Trees** | Sequential ensemble with boosting |

---

## 📈 Sample Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.78 | 0.76 | 0.79 | 0.77 | 0.82 |
| Decision Tree | 0.72 | 0.71 | 0.73 | 0.72 | 0.75 |
| Random Forest | 0.80 | 0.79 | 0.81 | 0.80 | 0.85 |
| SVM (LinearSVC) | 0.77 | 0.75 | 0.78 | 0.76 | 0.81 |
| **GBT** | **0.82** | **0.81** | **0.83** | **0.82** | **0.87** |

*Note: Results may vary based on train-test split*

---

## 🔧 Configuration

Modify the `CONFIG` dictionary in `Bank_Loan_Pro.py` to customize:

```python
CONFIG = {
    "dataset_path": "loan_data.csv",     # Dataset location
    "train_ratio": 0.8,                   # Training set ratio
    "random_seed": 42,                    # For reproducibility
    "rf_num_trees": 100,                  # Random Forest trees
    "gbt_max_iter": 20,                   # GBT iterations
    ...
}
```

---

## 📊 Visualizations Generated

1. **Loan Status Distribution** - Class balance analysis
2. **Income by Status** - Box plot of income vs approval
3. **Credit History Impact** - Approval rate by credit history
4. **Confusion Matrices** - For each model
5. **ROC Curves** - All models compared
6. **Model Comparison** - Bar chart of all metrics
7. **Feature Importance** - For tree-based models

---

## 🛠️ Technologies Used

- **Apache Spark (PySpark)** - Distributed data processing
- **Spark MLlib** - Machine learning at scale
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - ROC curve computation

---

## 📝 Future Improvements

- [ ] Add cross-validation for robust evaluation
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Add SHAP values for model explainability
- [ ] Create a Flask/Streamlit web interface
- [ ] Deploy model as REST API
- [ ] Add support for real-time predictions

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Arya Yadav**

---

