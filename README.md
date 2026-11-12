# Student Performance Prediction and Analysis System

A machine learning application that predicts student academic performance using Decision Tree Classification.

## Features

- **Dataset Management** — Upload CSV/Excel datasets or use the built-in sample dataset
- **Performance Prediction** — Predict student performance categories (Excellent, Good, Average, Needs Improvement)
- **Academic Risk Analysis** — Identify at-risk students based on input factors
- **Decision Tree Explorer** — Visualize the trained decision tree with node details
- **Model Evaluation** — View accuracy, precision, recall, F1 score, and confusion matrix
- **What-If Simulator** — Test how changing input values affects predictions
- **Recommendations** — Get personalized academic improvement suggestions
- **Prediction History** — Track and compare past predictions
- **PDF Reports** — Download professional performance reports

## Tech Stack

- Python
- Streamlit
- Scikit-learn (Decision Tree Classifier)
- Pandas / NumPy
- Matplotlib
- SQLite
- ReportLab (PDF generation)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Main application entry point
├── ml_engine.py            # Decision Tree ML engine
├── ml_pipeline.py          # ML pipeline utilities
├── data_manager.py         # Dataset upload and validation
├── auth.py                 # Database and prediction history
├── styles.py               # UI theme
├── pdf_report_generator.py # PDF report generation
├── views/                  # Application pages
│   ├── overview.py
│   ├── dataset_management.py
│   ├── prediction.py
│   ├── risk_analysis.py
│   ├── tree_explorer.py
│   ├── model_evaluation.py
│   ├── what_if_simulator.py
│   ├── recommendations.py
│   ├── history.py
│   └── about.py
├── advanced_student_data.csv  # Sample dataset
└── requirements.txt
```
