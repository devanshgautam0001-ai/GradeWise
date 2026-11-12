# Viva Preparation Guide: Student Performance Prediction and Analysis System

Comprehensive questions and technical answers for internal and external college viva evaluation.

---

## 1. Project Background & System Architecture

### Q1: What is the main objective of your project?
**Answer:**
The objective of the **Student Performance Prediction and Analysis System** is to provide an interpretable machine learning platform that early-detects academic distress in students, forecasts semester performance categories (`Excellent`, `Good`, `Average`, `Needs Improvement`), and provides actionable recommendations to prevent course failure. Unlike black-box models, it provides transparent, auditable decision paths using a Decision Tree Classifier.

### Q2: Why did you choose a Decision Tree Classifier instead of Random Forest, Neural Networks, or SVM?
**Answer:**
1. **Explainability & Transparency**: In educational and institutional contexts, students and advisors must understand *why* a particular performance or risk tier was assigned. A Decision Tree provides exact boolean rules (e.g., `If attendance <= 72% and study_hours <= 4.0 -> High Risk`).
2. **Auditable Decision Boundaries**: Faculty can verify and validate split thresholds against university policies.
3. **No Arbitrary Black-Box Inferences**: Deep learning or SVM models lack direct hierarchical rule extraction, making them inappropriate when accountability is required.

### Q3: What is the overall architecture of your application?
**Answer:**
The application follows a modular four-layer architecture:
1. **Presentation Layer**: Built with Streamlit, applying a custom dark navy/charcoal responsive CSS theme.
2. **Data Management & Ingestion Layer**: Handles CSV and Excel (.xlsx) file uploads, schema detection, missing value and duplicate validation, and preview statistics.
3. **Machine Learning Pipeline Layer**: Uses Scikit-Learn `Pipeline` and `ColumnTransformer` for leakage-free imputation, scaling, one-hot encoding, and Decision Tree training and inference.
4. **Data Persistence & Security Layer**: Local SQLite database storing user credentials encrypted with salted PBKDF2-HMAC-SHA256, alongside an isolated prediction audit trail.

---

## 2. Machine Learning & Decision Tree Fundamentals

### Q4: How does a Decision Tree decide where to split a node?
**Answer:**
At every node, the algorithm evaluates all candidate features and all possible split thresholds. It computes the **impurity reduction** (reduction in Gini Impurity or increase in Information Gain) for each candidate split:
$$\Delta I = I(\text{Parent}) - \left( \frac{N_{\text{Left}}}{N} I(\text{Left}) + \frac{N_{\text{Right}}}{N} I(\text{Right}) \right)$$
The feature and threshold that produces the maximum impurity reduction $\Delta I$ is selected as the split condition for that node.

### Q5: What is the mathematical definition of Gini Impurity and Shannon Entropy?
**Answer:**
- **Gini Impurity**: Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution:
  $$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$
  *Range*: From `0.0` (pure node, all samples belong to one class) to `1 - 1/C` (maximally impure).
- **Shannon Entropy**: Measures the degree of disorder or unpredictability in the node:
  $$\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
  *Range*: From `0.0` (pure node) to $\log_2(C)$ (uniform class distribution).

### Q6: What is Information Gain?
**Answer:**
Information Gain is the expected reduction in entropy achieved by partitioning samples on attribute $A$:
$$\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
The attribute with the highest Information Gain is selected to minimize uncertainty.

### Q7: What are the main hyperparameters in your Decision Tree, and how do they prevent overfitting?
**Answer:**
- `max_depth`: Limits the maximum depth of the tree from root to leaf, preventing the tree from carving out tiny partitions around noise.
- `min_samples_split`: The minimum number of samples an internal node must possess before it is permitted to split.
- `min_samples_leaf`: The minimum number of samples required to form a terminal leaf node. This ensures that every leaf represents a statistically meaningful group of students.
- `criterion`: Choice of impurity measurement (`gini` or `entropy`).

---

## 3. Data Preprocessing & Validation

### Q8: What is "Data Leakage", and how does your project prevent it?
**Answer:**
**Data leakage** occurs when information from the test dataset is inadvertently shared with the training process (for example, computing the mean or median of the entire dataset before splitting).
In our project:
- The dataset is split into `X_train` and `X_test` *first*.
- The `ColumnTransformer` (incorporating `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`) is fitted **strictly on `X_train`** using `.fit_transform()`.
- The holdout `X_test` is only transformed using `.transform()` with the statistics learned from `X_train`. This guarantees zero data leakage.

### Q9: How do you handle missing values and categorical data in your pipeline?
**Answer:**
- **Numerical Features**: Imputed using `SimpleImputer(strategy='median')` because the median is robust to outliers and skewed distributions, followed by `StandardScaler()`.
- **Categorical Features**: Imputed using `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. Setting `handle_unknown='ignore'` ensures that if an unseen category appears during inference, the pipeline does not throw an exception.

### Q10: How does your system handle continuous target scores like `final_score`?
**Answer:**
A Decision Tree Classifier requires discrete target classes. If the user selects a continuous score column, our system converts it into standardized performance categories using configurable, documented score ranges:
- Score $\ge 80.0$: `Excellent`
- $65.0 \le \text{Score} < 80.0$: `Good`
- $50.0 \le \text{Score} < 65.0$: `Average`
- $\text{Score} < 50.0$: `Needs Improvement`
Users can also adjust these thresholds directly in the Dataset Management interface.

---

## 4. Evaluation Metrics & Risk Analysis

### Q11: Why is Accuracy alone insufficient for evaluating multiclass student performance models?
**Answer:**
Educational cohorts often exhibit **class imbalance** (e.g., fewer students in `Needs Improvement` than in `Good` or `Average`). In such cases, a naive model could achieve high accuracy while failing to detect at-risk students.
Therefore, we compute:
- **Weighted Precision**: Measures how many of the students predicted to be in a category actually belonged to it.
- **Weighted Recall**: Measures the model's sensitivity in identifying all students in a given category.
- **Weighted F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: Identifies exact false positives and false negatives across all classes.

### Q12: How is Academic Risk determined? Is it generated randomly?
**Answer:**
Academic Risk is **never generated randomly**. It is determined through a two-stage deterministic policy:
1. The **Decision Tree Classifier** outputs the predicted performance category and class probabilities.
2. An **Institutional Risk Policy Engine** evaluates the prediction alongside critical academic indicators:
   - If predicted `Needs Improvement` $\rightarrow$ **High Risk**.
   - If predicted `Average` with attendance $< 70\%$ or study hours $< 3.0$ hrs $\rightarrow$ **High Risk**.
   - If predicted `Average` with stable attendance $\rightarrow$ **Medium Risk**.
   - If predicted `Good` with marginal attendance ($< 65\%$) $\rightarrow$ **Medium Risk**.
   - If predicted `Excellent` or `Good` with normal attendance $\rightarrow$ **Low Risk**.

---

## 5. Security, Software Engineering & Deliverables

### Q13: How is user authentication and prediction data secured?
**Answer:**
- Passwords are encrypted using **PBKDF2-HMAC-SHA256** with a unique 16-byte random salt per user and 100,000 iterations.
- If a legacy account logs in with an old plain SHA-256 hash, the system validates and automatically upgrades the stored hash to salted PBKDF2 without user friction.
- User prediction history is isolated in local SQLite using parameterized queries (`WHERE username = ?`), ensuring complete privacy.

### Q14: How does the What-If Simulator work?
**Answer:**
The What-If Simulator holds all student baseline inputs constant while sweeping a selected continuous variable (e.g., daily study hours from 1.0 to 12.0 hrs) across 25 intervals. It feeds each state through the active Decision Tree pipeline and plots the multi-class probability curves, showing students the exact break-even hours needed to transition to a higher performance tier.

### Q15: How is the PDF performance report generated?
**Answer:**
The PDF report is generated programmatically in-memory using **ReportLab Platypus**. It compiles student inputs, predicted categories, class probabilities, Decision Tree feature importance rankings, personalized recommendations, and institutional disclaimers into a downloadable document suitable for collegiate advising.

