# 🏦 Beta Bank Churn Prediction: From Data to Deployment

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://beta-bank-churn-prediction-carvalholevis.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[🚀 Explore the Interactive App](https://beta-bank-churn-prediction-carvalholevis.streamlit.app/)**  
**[🐍 View Source Code](https://github.com/carvalholevis/beta-bank-churn-prediction)**

</div>

---

## 🎯 The Mission

**Beta Bank** is facing a critical challenge: increasing customer churn.

Since retaining existing customers is significantly cheaper than acquiring new ones, the goal was to build a robust machine learning model to predict which customers are likely to leave, enabling proactive retention strategies.

This project is a full end-to-end data science solution, from exploratory analysis and model optimization to a live, interactive demo.

---

## 📊 Results

The final optimized **Random Forest model** exceeded the business target:

| Metric       | Target  | Achieved |
|--------------|---------|----------|
| **F1-Score** | > 0.59  | **0.61 ✅** |
| **AUC-ROC**  | —       | **0.85** |

---

## 🧠 Data Science Workflow

### 1. 🔎 Exploratory Data Analysis (EDA) & Data Preparation

- Handled missing values in `Tenure` using median
- Removed irrelevant features:
  - `RowNumber`
  - `CustomerId`
  - `Surname`
- Applied preprocessing:
  - One-Hot Encoding (categorical)
  - StandardScaler (numerical)
- Addressed class imbalance:
  - Used **Upsampling**
  - More effective than `class_weight='balanced'`

---

### 2. 🤖 Model Development & Optimization

- Algorithm: **Random Forest Classifier**
- Hyperparameter tuning:
  - Used `RandomizedSearchCV`
  - 5-fold cross-validation
  - Tested 100+ combinations
- Evaluation metrics:
  - Primary: **F1-score**
  - Secondary: **AUC-ROC**

---

### 3. 🚀 Deployment

- Modular project structure (`src/`)
- Model saved using `joblib`
- Interactive app built with **Streamlit**
- Deployed on **Streamlit Community Cloud**

---

## 🗂️ Project Structure

```
beta-bank-churn-prediction/
├── config.py
├── data/
│   └── Churn.csv
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── 01_beta_bank_analysis_and_modeling.ipynb
├── src/
│   ├── utils.py
│   ├── data_preparation.py
│   ├── model_training.py
│   └── evaluation.py
├── app.py
├── train_and_save.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/carvalholevis/beta-bank-churn-prediction.git
cd beta-bank-churn-prediction
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Windows:

```
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Download dataset

Download the dataset and place it inside the `data/` folder:

[![Download Dataset](https://img.shields.io/badge/Dataset-Download-blue?style=for-the-badge&logo=databricks&logoColor=white)](https://practicum-content.s3.us-west-1.amazonaws.com/datasets/Churn.csv)

### 5. Train the model

```
python train_and_save.py
```

### 6. Run the app

```
streamlit run app.py
```

---

## 👨‍💻 About the Author

**Leviton Lima Carvalho**

- 💼 LinkedIn: https://www.linkedin.com/in/levitoncarvalho/
- 🐙 GitHub: https://github.com/carvalholevis
- ✉️ Email: carvalholevis@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
