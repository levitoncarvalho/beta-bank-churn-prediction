# 🏦 Beta Bank Churn Prediction
### *Turning Behavioral Data into Intelligent Customer Retention*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://beta-bank-churn-prediction-carvalholevis.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br>

**[🚀 Try the Live App](https://beta-bank-churn-prediction-carvalholevis.streamlit.app/)** &nbsp;|&nbsp; **[📓 View Full Notebook](notebooks/01_beta_bank_analysis_and_modeling.ipynb)**

</div>

---

> ⚠️ **Disclaimer:** Beta Bank is a **fictional institution** created exclusively for academic and portfolio purposes. This project was developed as part of a Data Science training program and is intended solely to demonstrate technical skills in machine learning, data analysis, and model deployment. No real financial institution, customer data, or business relationship is represented here.

---

## 🧩 The Business Problem

> *"Acquiring a new customer costs 5 to 25 times more than retaining an existing one."*
> — Harvard Business Review

**Beta Bank** was facing a silent but devastating problem: gradual customer churn. Month after month, clients were leaving — and the business team only noticed after the damage was done.

The strategic question was clear: **how do we act *before* a customer leaves?**

The answer: build a machine learning model capable of identifying, in advance, which customers are at the highest risk of churning — enabling the bank to take proactive, personalized retention actions.

### Why Does This Matter?

| Scenario | Impact |
|---|---|
| 🔴 No predictive model | The bank reacts *after* losing the customer |
| 🟡 Weak model (F1 < 0.59) | Too many false negatives — at-risk customers go undetected |
| 🟢 **Optimized model (F1 = 0.61, AUC-ROC = 0.85)** | **Reliable early identification of at-risk customers** |

---

## 📊 Final Results

<div align="center">

### 🏆 The final model exceeded the business target

</div>

| Metric | Target | Result | Status |
|---|---|---|---|
| **F1-Score** (test set) | > 0.59 | **0.61** | ✅ Exceeded |
| **AUC-ROC** (test set) | Benchmark | **0.85** | ✅ Strong |
| **Algorithm** | — | Random Forest | — |
| **Imbalance Technique** | — | Upsampling | — |

> **An AUC-ROC of 0.85 means the model correctly distinguishes a churning customer from a retained one 85% of the time** — highly effective discrimination for targeted retention campaigns.

---

## 🧠 End-to-End Data Science Workflow

```
📥 Raw Data  →  🔎 EDA  →  🛠️ Preprocessing  →  ⚖️ Class Balancing  →  🤖 Modeling  →  🎯 Evaluation  →  🚀 Deployment
```

### 1. 🔎 Exploratory Data Analysis (EDA)

Before any modeling, the data was thoroughly investigated:

- **10,000 customer records** with 13 behavioral and demographic features
- **Missing values in `Tenure`** (~2.7% of records) → imputed using median
- Columns with no predictive value removed: `RowNumber`, `CustomerId`, `Surname`
- **Critical finding:** the dataset is heavily imbalanced — only ~20.4% of customers churned (`Exited = 1`), which distorts simple accuracy metrics and demands specialized treatment

**Key patterns discovered in EDA:**
- Older customers churn significantly more
- Customers with higher balances show elevated churn — suggesting migration to competitors with better rates
- Customers in **Germany** churn at a much higher rate than France or Spain
- **Inactive members** are far more likely to leave
- Customers using **exactly 2 products** are the most loyal; 1 or 3-4 products correlate with higher churn

**Feature Overview:**

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Numerical | Customer credit score |
| `Geography` | Categorical | Country of residence |
| `Gender` | Categorical | Gender |
| `Age` | Numerical | Customer age |
| `Tenure` | Numerical | Years as a bank customer |
| `Balance` | Numerical | Account balance |
| `NumOfProducts` | Numerical | Number of banking products used |
| `HasCrCard` | Binary | Owns a credit card (1=Yes, 0=No) |
| `IsActiveMember` | Binary | Active member (1=Yes, 0=No) |
| `EstimatedSalary` | Numerical | Estimated annual salary |

---

### 2. ⚖️ Handling Class Imbalance

One of the core challenges of this project was the **class imbalance**. Two approaches were tested and compared:

| Approach | F1-Score (Validation) | Notes |
|---|---|---|
| No balancing (baseline) | ~0.50 | Model biased toward the majority class |
| `class_weight='balanced'` | ~0.57 | Improvement, but still below target |
| **Upsampling (oversampling minority class)** | **~0.61** | **Best result — chosen technique** |

**Why upsampling won:** Replicating minority-class examples in the training set forced the model to learn richer, more varied patterns from customers who actually churned — leading to better generalization on unseen data.

---

### 3. 🤖 Modeling & Optimization

- **Algorithm:** `RandomForestClassifier` — chosen for its robustness with tabular data, resistance to overfitting, and interpretability via feature importance
- **Optimization:** `RandomizedSearchCV` with 5-fold cross-validation, exploring 100+ hyperparameter combinations
- **Tuned parameters:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Primary metric:** F1-Score | **Secondary metric:** AUC-ROC

---

### 4. 🎯 Final Test Set Evaluation

The final model was evaluated on data **never seen during training**:

```
F1-Score  = 0.61  ✅  (target: > 0.59)
AUC-ROC   = 0.85  ✅
```

The ROC curve with AUC = 0.85 demonstrates strong discriminative power — the model reliably separates customers who will leave from those who will stay, across all classification thresholds.

---

### 5. 🚀 Deployment

The trained model was serialized with `joblib` and deployed as an interactive **Streamlit web app**, allowing the bank's team to input customer data and receive real-time churn predictions.

**[➡️ Try the live app here](https://beta-bank-churn-prediction-carvalholevis.streamlit.app/)**

---

## 🗂️ Project Structure

```
beta-bank-churn-prediction/
│
├── 📓 notebooks/
│   └── 01_beta_bank_analysis_and_modeling.ipynb   # Full analysis + modeling
│
├── 🤖 models/
│   ├── best_model.pkl                              # Serialized trained model
│   └── scaler.pkl                                  # Serialized scaler
│
├── 🐍 src/
│   ├── data_preparation.py                         # Preprocessing pipeline
│   ├── model_training.py                           # Training & optimization
│   ├── evaluation.py                               # Metrics & visualizations
│   └── utils.py                                    # Helper functions
│
├── 🌐 app.py                                       # Streamlit interactive app
├── 🔁 train_and_save.py                            # Model retraining script
├── ⚙️ config.py                                    # Centralized configuration
├── 📦 requirements.txt
└── 📄 README.md
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/carvalholevis/beta-bank-churn-prediction.git
cd beta-bank-churn-prediction

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 5. Train the model
python train_and_save.py

# 6. Launch the app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib |
| **Deployment** | Streamlit, Streamlit Community Cloud |

---

## 💡 Key Technical Takeaways

- **Class imbalance is a real threat to model quality** — simple accuracy is a misleading metric when classes are disproportionate. F1-Score and AUC-ROC are far more appropriate for churn prediction.
- **Upsampling outperformed `class_weight`** in this scenario — generating additional minority-class examples allowed the model to learn more diverse churning patterns.
- **Random Forest + RandomizedSearchCV** proved to be an efficient combination: broad hyperparameter space exploration without excessive computational cost.
- The **modular code structure** (`src/`) promotes maintainability, testability, and scalability — a best practice for production ML systems.

---

## 👨‍💻 Author

<div align="center">

**Leviton Lima Carvalho**
*Data Scientist | Machine Learning | Python*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-levitoncarvalho-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/levitoncarvalho/)
[![GitHub](https://img.shields.io/badge/GitHub-levitoncarvalho-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/levitoncarvalho)
[![Email](https://img.shields.io/badge/Email-levitoncarvalho@icloud.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:levitoncarvalho@icloud.com)

</div>

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.
