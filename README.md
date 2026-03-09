# Dark Pattern Risk Detection System
### AI-Powered Detection of Manipulative UX Design Using Machine Learning

## Overview

Dark Patterns are deceptive design techniques used in websites and applications to manipulate users into making unintended decisions such as unwanted subscriptions, hidden purchases, or misleading clicks.

This project builds a **Machine Learning based Dark Pattern Detection System** that analyzes **user behavioral session data** and predicts the **probability of manipulative design patterns**.

The system combines:

- Machine Learning (Random Forest)
- Behavioral Data Analysis
- Streamlit Web Application
- Interactive Analytics Dashboard

The goal is to help **UX Designers, Data Analysts, and Compliance Teams detect unethical user interface patterns.**

---

# Project Architecture

```

User Behavior Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Machine Learning Model (Random Forest)
↓
Risk Score Prediction
↓
Streamlit Web Application
↓
Interactive Analytics Dashboard

```

---

# Key Features

- Dark Pattern Risk Prediction
- Behavioral Pattern Analysis
- Real-time Risk Scoring
- Interactive Streamlit Dashboard
- Bulk CSV Prediction
- Data Visualization with Plotly
- Feature Importance Analysis

---

# Dataset

Dataset Used:

Online Shoppers Purchasing Intention Dataset

Source:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Dataset Information:

- Total Records: 12,330 sessions
- Features: 18
- Numerical Features: 10
- Categorical Features: 8

Key Behavioral Features:

- Bounce Rate
- Exit Rate
- Page Value
- Session Duration
- Visitor Type
- Product Related Interaction

---

# Machine Learning Models Used

Multiple algorithms were evaluated:

- Logistic Regression
- Support Vector Machine
- K Nearest Neighbors
- Gradient Boosting
- Random Forest (Best Performing)

Final Model Performance:

Accuracy: **98.2%**  
F1 Score: **0.93**

Random Forest was selected due to its strong performance and interpretability.

---

# Feature Importance

The most important behavioral indicators detected by the model:

1. PageValues
2. ExitRates
3. ProductRelated_Duration
4. BounceRates

These signals reveal **user frustration patterns and potential manipulation loops.**

---

# Streamlit Application

The project includes a **fully interactive Streamlit web app** that allows users to:

Single User Prediction
- Enter behavioral data
- Get instant Dark Pattern Risk Score

Bulk CSV Prediction
- Upload datasets
- Analyze multiple sessions
- Generate risk reports

Run the app:

```bash
streamlit run streamlit_app.py
```

---

# Dashboard Analytics

The dashboard provides insights including:

- Risk Score Distribution
- High Risk User Sessions
- Visitor Type Risk Analysis
- Exit vs Bounce Correlation
- Monthly Risk Trends

These analytics help identify **when and where manipulative UX patterns occur.**

---

# Project Structure

```
Dark-Pattern-Risk-Detection-System
│
├── dataset
├── models
├── app
├── notebooks
├── dashboard
├── images
├── requirements.txt
└── README.md
```

---

# Installation


```

Run the Streamlit app

streamlit run streamlit_app.py
```

---

# Results

Model Accuracy: **98%**

High Risk Detection Recall: **89%**

The system successfully identifies behavioral patterns that indicate:

- Manipulative navigation loops
- Hidden cost flows
- Forced action UX designs

---

# Future Improvements

Possible enhancements include:

- Computer Vision for detecting visual dark patterns
- Browser Extension for real-time detection
- Deep Learning based behavior analysis
- Mobile UX pattern detection
- Real-time website monitoring

---

# Author

Adithya D  
Data Analyst

---

# License

This project is open-source and available under the MIT License.
