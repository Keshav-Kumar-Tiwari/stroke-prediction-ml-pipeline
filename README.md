# 🧠 Stroke Prediction ML Pipeline

An end-to-end **Machine Learning project** to predict the likelihood of stroke using healthcare data, with a strong focus on handling **class imbalance** and improving **stroke detection (recall)**.

---

## 📌 Problem Statement

Stroke prediction is a critical healthcare problem where detecting positive cases is more important than simply achieving high overall accuracy.

The dataset is **highly imbalanced**, with stroke cases being much fewer than non-stroke cases.

A model trained without handling this imbalance may achieve high accuracy while failing to identify actual stroke patients.

Therefore, this project focuses on:

* Handling class imbalance
* Improving **Recall for stroke cases**
* Performing meaningful feature engineering
* Building a clean and reusable ML pipeline

---

## 📊 Dataset

**Source:** Kaggle — Stroke Prediction Dataset

🔗 [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Dataset Characteristics

The dataset contains healthcare-related information such as:

* Gender
* Age
* Hypertension
* Heart disease
* Marital status
* Work type
* Residence type
* Average glucose level
* BMI
* Smoking status

### Target Variable

`stroke`

| Value | Meaning   |
| ----- | --------- |
| `0`   | No Stroke |
| `1`   | Stroke    |

The target variable is highly imbalanced, making **recall, precision, F1-score, and confusion matrix** important evaluation metrics.

---

## 🔍 Exploratory Data Analysis (EDA)

Several important patterns were identified during EDA:

### 📈 Age

Age was one of the strongest predictors of stroke risk. Stroke cases were considerably more common among older individuals.

### ⚠️ Class Imbalance

The dataset contains significantly fewer stroke cases than non-stroke cases.

This makes accuracy alone an unreliable metric.

### 📉 Average Glucose Level

`avg_glucose_level` showed a skewed distribution.

A **log transformation** was applied to reduce the effect of extreme values.

### 🚬 Smoking Status

Several records contained an `Unknown` smoking status.

Instead of simply treating these values as a normal category, an additional feature was created:

```text
is_smoking_unknown
```

### 🏠 Residence Type

`Residence_type` showed limited predictive value and was removed from the final feature set.

---

## ⚙️ Machine Learning Approach

The project follows a structured ML workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
Train/Test Split
     ↓
Logistic Regression
     ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

The following preprocessing techniques were used:

* Missing value imputation
* Handling categorical variables
* One-hot encoding
* Feature scaling
* Log transformation of skewed features
* Removal of unnecessary features

A **Scikit-learn Pipeline** was used to combine preprocessing and model training into a single workflow.

---

## 🛠️ Feature Engineering

### `is_smoking_unknown`

A new binary feature was created to capture whether the smoking status was unknown.

```python
df["is_smoking_unknown"] = (
    df["smoking_status"] == "Unknown"
).astype(int)
```

This allows the model to distinguish between:

* Known smoking information
* Missing/unknown smoking information

---

## ⚖️ Handling Class Imbalance

Because stroke cases are rare, a standard Logistic Regression model struggled to identify positive cases.

### ❌ Initial Model

The initial model did not properly handle class imbalance.

Result:

```text
Stroke Recall = 0.00
```

The model failed to detect stroke cases.

### ✅ Class-Weighted Logistic Regression

To address this problem, **class weighting** was introduced:

```python
LogisticRegression(class_weight="balanced")
```

This gives greater importance to the minority class during training.

### Why Class Weight?

Instead of artificially generating new samples, the model is instructed to pay more attention to the underrepresented stroke class.

This project intentionally avoided oversampling methods such as **SMOTE**.

---

## 🤖 Model

### Logistic Regression

The primary model used in this project is:

**Logistic Regression with class balancing**

Advantages:

* Simple and interpretable
* Fast to train
* Works well as a baseline classification model
* Provides probability estimates
* Easy to integrate with Scikit-learn pipelines

---

## 📈 Results

### Before Class Imbalance Handling

```text
Stroke Recall: 0.00
```

The model failed to identify stroke cases.

### After Class Weighting

```text
Stroke Recall: 0.84
```

The balanced Logistic Regression model successfully detected the majority of stroke cases.

### ⚠️ Trade-off

Improving recall resulted in lower precision.

This means the model identifies more actual stroke cases but also produces more **false positives**.

This is an important trade-off in an imbalanced classification problem.

---

## 📊 Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Why Recall?

For this problem, **Recall is particularly important** because failing to identify a potential stroke case can be more costly than generating an additional false positive.

Recall is calculated as:

```text
Recall = True Positives / (True Positives + False Negatives)
```

---

## 🧠 Key Learnings

Through this project, I learned:

* How to work with real-world healthcare data
* Why accuracy can be misleading for imbalanced datasets
* How to perform Exploratory Data Analysis
* How to handle missing values
* How to encode categorical variables
* How feature engineering can improve model performance
* How to handle class imbalance using `class_weight`
* How to use Scikit-learn Pipelines
* Why recall can be more important than accuracy in certain classification problems
* How improving recall can increase false positives

---

## 🚀 Future Improvements

The project can be further improved by experimenting with:

### 🌲 Advanced Models

* Random Forest
* XGBoost
* Gradient Boosting
* Support Vector Machine

### 🎯 Threshold Tuning

The default classification threshold can be adjusted to find a better balance between:

* Precision
* Recall
* F1-score

### 📊 Model Comparison

Compare multiple models using the same evaluation metrics.

### 🌐 Deployment

Deploy the trained model as a web application using:

* Flask
* FastAPI
* Streamlit

### 📈 Explainability

Add model explainability techniques such as:

* Feature importance
* SHAP
* Model coefficients

---

## 🛠️ Tech Stack

### Programming Language

* 🐍 Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Data Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* VS Code

---

## 📂 Project Structure

```text
Stroke-Prediction-ML/
│
├── 📁 data/
│   └── healthcare-dataset-stroke-data.csv
│
├── 📁 notebooks/
│   └── stroke_prediction.ipynb
│
├── 📁 src/
│   └── model.py
│
├── 📁 models/
│   └── stroke_model.pkl
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore
```

> The project structure can be adjusted depending on how the final implementation is organized.

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stroke-prediction-ml.git
```

### 2. Navigate to the Project

```bash
cd stroke-prediction-ml
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

Open:

```text
notebooks/stroke_prediction.ipynb
```

and run the cells sequentially.

---

## 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## ⚠️ Disclaimer

This project is created for **educational and machine-learning practice purposes only**.

It is **not a medical diagnostic system** and should not be used to make real-world medical decisions.

---

## 👨‍💻 Author

**Keshav Kumar Tiwari**

B.Tech — Information Technology

---

⭐ If you found this project useful, consider giving the repository a **star**!
