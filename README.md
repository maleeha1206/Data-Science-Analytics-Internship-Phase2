# Data Science & Machine Learning Internship - Phase 2 Portfolio

Welcome to the second phase of my Data Science portfolio repository. While Phase 1 established foundational predictive structures, Phase 2 implements advanced enterprise machine learning workflows, including Explainable AI (XAI), Unsupervised Clustering spaces, and comparative multi-model Time Series forecasting.

## 📁 Repository Structure
* 📝 `Phase2 Task1.ipynb` - Classification with Random Forests and SHAP interpretability.
* 📝 `Phase2 Task2.ipynb` - Unsupervised K-Means clustering and PCA dimensionality reduction.
* 📝 `Phase2 Task3.ipynb` - Time Series Forecasting: Temporal predictive comparison across ARIMA, Prophet, and XGBoost.

---

## 🛠️ Project Deep Dives

### Task 1: Bank Term Deposit Prediction (Explainable AI)
* **File:** `Phase2 Task1.ipynb`
* **Objective:** Predict if a customer will open a term deposit and decode model parameters using XAI frameworks.
* **Architecture:** Random Forest Classifier & Logistic Regression baseline (`saga` optimizer).
* **Interpretability Framework:** Integrated **SHAP (SHapley Additive exPlanations)** to map explicit local and global feature impact parameters, proving that telemarketing call `duration` and historical `poutcome_success` are the strongest drivers for customer conversion.
* **Metrics Achieved:** Full evaluation via Confusion Matrix, F1-Score, and Receiver Operating Characteristic (ROC-AUC) curves.

### Task 2: Behavioral Customer Segmentation (Unsupervised Learning)
* **File:** `Phase2 Task2.ipynb`
* **Objective:** Group retail shoppers based on high-dimensional spending patterns to formulate marketing strategies.
* **Workflow:** Scaled raw metrics using `StandardScaler` to remove distance skewing. Applied the **Elbow Method** to calculate the Within-Cluster Sum of Squares (WCSS), identifying $K=5$ as the optimal grouping point.
* **Visualization:** Implemented **Principal Component Analysis (PCA)** to reduce the 3D metric landscape down to a 2D space for cluster mapping.

### Task 3: Energy Consumption Time Series Forecasting 
* **File:** `Phase2 Task3.ipynb`
* **Objective:** Construct an evaluation pipeline to forecast short-term power distribution needs.
* **Feature Engineering:** Extracted cyclical time-index vectors (`Month`, `WeekOfYear`, `Quarter`) to enable tabular machine learning ingestion.
* **Comparative Matrix:** Evaluated three distinct predictive core methodologies:
  1. *ARIMA(1,1,1)* (Classical Autoregressive Statistical Model)
  2. *Meta Prophet* (Additive Seasonal/Yearly Growth Framework)
  3. *XGBoost Regressor* (Gradient-Boosted Decision Trees)
* **Evaluation Metrics:** Quantified model residual error values through Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## 💻 Technical Stack
* **Language:** Python 3.13
* **Core Libraries:** Scikit-Learn, SHAP, Prophet, XGBoost, Statsmodels, Pandas, NumPy, Matplotlib, Seaborn.
