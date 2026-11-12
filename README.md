# Student Performance Prediction and Analysis System

An institutional-grade collegiate Machine Learning analytics platform designed to forecast student academic outcomes, stratify academic risk, visualize decision tree logic, and deliver evidence-based curricular intervention recommendations.

Built with **Python**, **Streamlit**, **Scikit-Learn**, **Pandas**, **NumPy**, **Matplotlib**, **ReportLab**, and **SQLite3**.

---

## 1. Project Introduction

The **Student Performance Prediction and Analysis System** provides higher-education institutions, academic mentors, and students with an interpretable machine learning dashboard. Unlike black-box models, this platform employs a pure **Decision Tree Classifier (CART)**, allowing educators and students to audit the exact hierarchical rules, split thresholds, and feature contributions that lead to a specific academic performance category.

The platform eliminates arbitrary manual heuristics and third-party AI APIs, operating 100% locally with verifiable data privacy and salted PBKDF2 cryptographic user authentication.

---

## 2. Problem Statement

In contemporary higher education:
1. **Late Interventions**: Academic distress is typically discovered only after semester grades or midterms are published—too late for timely course correction.
2. **High Student-to-Faculty Ratios**: Academic counselors cannot manually audit every student's daily study habits, attendance percentage, stress levels, and assignment completion.
3. **Lack of Explainability**: Modern deep learning or complex ensemble models often fail in institutional settings because students and advisors cannot interpret *why* a particular grade or risk status was assigned.

This system resolves these challenges by providing an early-warning diagnostic platform that couples interpretable Decision Tree ML with deterministic institutional risk policies.

---

## 3. Project Objectives

- **Automated Data Ingestion**: Accept custom university cohort records via CSV or Excel (.xlsx) formats with full data hygiene and validation checks.
- **Leakage-Free ML Preprocessing**: Implement Scikit-Learn `ColumnTransformer` pipelines that impute, scale, and encode features strictly on training data splits.
- **Interpretable Classification**: Train genuine Decision Tree Classifiers mapping student features into standardized performance categories (`Excellent`, `Good`, `Average`, `Needs Improvement`).
- **Interactive Tree Visualizer**: Render graphical tree structures, split thresholds, Gini impurities, and text-based decision paths.
- **Transparent Academic Risk Stratification**: Apply documented academic policies that flag Low, Medium, and High Risk students based on model classifications and critical threshold gates.
- **Scenario Simulation**: Allow students to simulate what-if study habit adjustments and inspect dynamic category transition probabilities.
- **Publication-Grade PDF Reporting**: Generate downloadable academic assessment diagnostic reports in PDF format for academic meetings.
- **Private Audit Trail**: Record student predictions in a local SQLite database isolated by authenticated user sessions.

---

## 4. Algorithm Explanation: Decision Tree Classifier

The primary classification algorithm utilized throughout this platform is the **Decision Tree Classifier** based on the **Classification and Regression Trees (CART)** algorithm.

A Decision Tree is a non-parametric supervised learning algorithm that partitions the feature space into orthogonal, axis-parallel hyper-rectangles. Each internal node represents a binary test on an attribute, each branch represents the outcome of the test, and each leaf node holds a probability distribution and majority class assignment.

### Core Mathematical Advantages:
- **Interpretability**: Every decision can be traced back to human-readable boolean conditional statements (e.g., `If Daily Study Hours <= 4.2 and Attendance <= 68% -> High Risk`).
- **Mixed Feature Support**: Effectively handles combinations of continuous numerical features (study hours, attendance) and categorical indicators.
- **No Linearity Assumption**: Captures non-linear thresholds and complex feature interactions without feature engineering.

---

## 5. Decision Tree Working Mechanism

1. **Root Node Selection**: Evaluates all candidate features and split thresholds across the training cohort to identify the single split that maximizes impurity reduction.
2. **Recursive Binary Splitting**: Partitions the subset into left and right child nodes and repeats the splitting process recursively.
3. **Stopping Criteria**:
   - `max_depth`: Limits the maximum distance between the root and terminal leaf nodes to avoid overfitting noisy samples.
   - `min_samples_split`: Enforces a minimum sample threshold before an internal node is permitted to split.
   - `min_samples_leaf`: Ensures that every terminal partition contains a sufficient number of students to generalize well.
4. **Leaf Node Assignment**:
   The predicted class for any sample reaching leaf node $L$ is determined by plurality voting:
   $$\hat{y} = \arg\max_{c} \left( \frac{N_{c, L}}{N_L} \right)$$
   Where $N_{c, L}$ is the count of training students in leaf $L$ belonging to class $c$, and $N_L$ is the total count in leaf $L$.

---

## 6. Entropy and Information Gain

The platform supports two classic split criteria:

### A. Gini Impurity (Default)
Gini impurity measures the frequency with which an element would be incorrectly labeled if randomly labeled according to the distribution of labels in the subset:
$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$
Where $C$ is the number of performance classes and $p_i$ is the relative frequency of class $i$ in node $S$. A node with a Gini score of `0.0` is purely homogeneous.

### B. Shannon Entropy & Information Gain
Shannon entropy measures the uncertainty or unpredictability of the target class distribution:
$$\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

The **Information Gain (IG)** of a candidate split on attribute $A$ represents the total reduction in entropy:
$$\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
The algorithm selects the feature $A$ and threshold $t$ that achieves the highest Information Gain.

---

## 7. Dataset Format Requirements

The system accepts tabular student performance datasets in **CSV (.csv)** or **Microsoft Excel (.xlsx, .xls)** formats.

### Recommended Features:
| Feature Column | Expected Type | Description |
| :--- | :--- | :--- |
| `study_hours` | Numeric (Float) | Average daily self-study hours (e.g., 0.0 - 14.0) |
| `attendance` | Numeric (Float) | Classroom attendance percentage (0.0 - 100.0%) |
| `sleep_hours` | Numeric (Float) | Nightly sleep duration (e.g., 4.0 - 10.0 hrs) |
| `previous_score` | Numeric (Float) | Previous exam or midterm score (0.0 - 100.0) |
| `social_media_usage`| Numeric (Float) | Daily entertainment screen time in hours |
| `stress_level` | Numeric (Int) | Perceived academic stress scale (1 - 10) |
| `preparation_level`| Numeric (Int) | Exam readiness self-assessment (1 - 5) |
| `extra_activities` | Numeric (Int) | Weekly extracurricular hours (0 - 25) |
| `final_score` or `performance_category` | Target | Numerical score or discrete class labels |

---

## 8. Dataset Upload Instructions

1. Navigate to the **Dataset Management** module in the sidebar.
2. Select **Upload Your Dataset**.
3. Use the Streamlit file uploader to browse your computer or drag and drop a `.csv` or `.xlsx` file.
4. Click **Apply & Activate Dataset**.
5. The automated validation engine verifies:
   - Format integrity (rejects unsupported extensions).
   - Non-zero file size (rejects empty 0-byte files).
   - Parsing consistency (checks delimiters and corrupt byte arrays).
   - Schema detection (identifies numerical vs. categorical columns).
   - Quality checks (missing values count & duplicate row count).

*Note: You may also select the built-in benchmark cohort (`advanced_student_data.csv`, 1,000 records) at any time.*

---

## 9. Data Preprocessing Pipeline

To eliminate data leakage, all preprocessing is encapsulated within Scikit-Learn `Pipeline` and `ColumnTransformer` constructs:
1. **Numerical Transformer**:
   - `SimpleImputer(strategy='median')`: Replaces any missing numerical values with training-set median values.
   - `StandardScaler()`: Standardizes features by removing the mean and scaling to unit variance.
2. **Categorical Transformer**:
   - `SimpleImputer(strategy='most_frequent')`: Imputes missing categorical entries with modal categories.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Encodes categorical attributes without failing if unseen categories emerge during prediction.
3. **Training Isolation**: The `ColumnTransformer` is fitted **strictly on `X_train`**.

---

## 10. Model Training Process

1. In **Dataset Management** or **Model Evaluation**, select the desired **Target Column**.
   - If the target is numeric (e.g., `final_score`), the system converts continuous scores into documented performance categories using customizable cutoffs:
     - $\ge 80.0$: `Excellent`
     - $65.0 - 79.9$: `Good`
     - $50.0 - 64.9$: `Average`
     - $< 50.0$: `Needs Improvement`
2. Select or deselect feature columns (excluding student IDs or irrelevant identifiers).
3. Tune Decision Tree hyperparameters:
   - **Splitting Criterion**: Gini vs. Entropy.
   - **Max Depth**: Limits tree depth (1 - 20).
   - **Min Samples Split**: Minimum samples required to split an internal node (2 - 50).
   - **Min Samples Leaf**: Minimum samples required in a terminal leaf (1 - 30).
   - **Test Size Ratio**: Holdout proportion (default 20%).
4. Click **Train Decision Tree Classifier**.
5. The pipeline executes a stratified train-test split (`stratify=y`), fits the model, and serializes the complete bundle into `decision_tree_model.pkl`.

---

## 11. Evaluation Metrics

Model performance is evaluated strictly on the unseen holdout test partition ($N = 200$ for 1,000 samples):
- **Accuracy**: Overall fraction of correctly categorized students.
- **Weighted Precision**: Proportion of correct positive predictions weighted by class support:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Weighted Recall**: Proportion of actual members of a performance tier successfully detected:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Weighted F1 Score**: Harmonic mean of precision and recall:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix**: Visual heatmap displaying actual vs. predicted category distributions.
- **Classification Report**: Full class-by-class breakdown of precision, recall, and support.

---

## 12. Installation Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 (compatible with Python 3.14)
- Git (optional)

### Steps
1. Clone or extract the project repository:
   ```bash
   cd "AI_study plannar"
   ```

2. Create and activate a clean virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## 13. Execution Instructions

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Open your browser to:
`http://localhost:8501`

### Default Login Credentials (or register a new account):
- **Username**: `gurpreet@456`
- **Password**: `1234`
*(New accounts can be registered instantly via the "Register Account" tab).*

---

## 14. Project Folder Structure

```
AI_study plannar/
│
├── app.py                     # Main application entrypoint & sidebar router
├── auth.py                    # SQLite database, PBKDF2 hashing, user audit log
├── data_manager.py            # Dataset upload, CSV/Excel parsing, validation, statistics
├── ml_engine.py               # Core Decision Tree ML backend, training, evaluation, inference
├── pdf_report_generator.py    # ReportLab publication-grade PDF report engine
├── styles.py                  # Premium dark navy CSS theme, metric cards, badges
├── requirements.txt           # Python library dependencies
├── users.db                   # Local SQLite database (users and predictions)
├── advanced_student_data.csv  # Verified 1,000-record benchmark cohort dataset
├── student_data.csv           # Introductory 10-record sample dataset
├── decision_tree_model.pkl    # Serialized Decision Tree model & preprocessing pipeline
│
├── views/                     # Modular dashboard views (10 sections)
│   ├── __init__.py            # Views package initialization
│   ├── overview.py            # Section 1: Executive overview & academic analytics
│   ├── dataset_management.py  # Section 2: Dataset upload, preview, validation, target setup
│   ├── prediction.py          # Section 3: Dynamic prediction, probabilities, PDF export
│   ├── risk_analysis.py       # Section 4: Academic risk stratification & vulnerability drivers
│   ├── tree_explorer.py       # Section 5: Decision tree graph, export_text, math theory
│   ├── model_evaluation.py    # Section 6: Hyperparameters, metrics, confusion matrix, report
│   ├── what_if_simulator.py   # Section 7: Parametric what-if study sensitivity curves
│   ├── recommendations.py     # Section 8: Rule-based personalized academic action plan
│   ├── history.py             # Section 9: Private SQLite history, trends, comparison, deletion
│   └── about.py               # Section 10: Technical documentation, architecture, limitations
│
├── README.md                  # Comprehensive project documentation
└── VIVA_QUESTIONS_AND_ANSWERS.md # Viva examination preparation document
```

---

## 15. Screenshots & Interface Walkthrough

1. **Sign-In Portal**: Dark charcoal interface with salted PBKDF2 password authentication and instant account registration.
2. **Executive Overview**: High-level KPIs, cohort dataset metrics, Decision Tree accuracy pill, and cohort score distributions.
3. **Dataset Management**: Drag-and-drop file uploader supporting CSV/XLSX, 10-row preview, missing values breakdown, and target configuration.
4. **Performance Prediction**: Dynamic input sliders adapted to dataset features, predicted performance category, probability distribution bars, and one-click PDF report download.
5. **Academic Risk Analysis**: Documented institutional risk tiers (Low, Medium, High Risk) with empirical cohort distributions.
6. **Decision Tree Explorer**: High-resolution `plot_tree` visual diagrams with depth zoom, hierarchical text rules, and mathematical equations.
7. **Model Evaluation**: Test Accuracy, F1 Score, Confusion Matrix heatmap with true labels, and classification report.
8. **What-If Simulator**: Interactive sensitivity simulator with multi-line class probability curves.
9. **Personalized Recommendations**: Categorized action cards organized by Critical, Habit, and Wellness priorities.
10. **Prediction History**: Searchable audit log, side-by-side comparison selector, performance trend charts, and CSV export.

---

## 16. Limitations and Future Scope

### Limitations
- **Orthogonal Splitting**: Standard Decision Trees make axis-parallel splits, which can approximate diagonal decision boundaries through staircase patterns.
- **Sensitivity to Training Variance**: Individual decision trees can be sensitive to minor dataset variations (controlled via `min_samples_split` and `min_samples_leaf`).
- **Educational Context**: Provides early-warning indicators and recommendations; does not replace holistic faculty counseling.

### Future Scope
- **Direct Learning Management System (LMS) Integration**: Automated webhooks to pull live attendance and assignment logs from Moodle, Canvas, or Blackboard.
- **Cost-Complexity Pruning ($\alpha$-pruning)**: Automatic hyperparameter optimization using cross-validated pruning paths.
- **Multimodal Student Engagement Tracking**: Incorporating forum participation and assignment submission timestamps.
