# Credit Risk Modeling & Lending Strategy Optimization

This project builds a machine learning pipeline to **predict credit card defaults** and optimize lending decisions based on **expected revenue** and **default risk**. It leverages both **XGBoost** and **Neural Networks** and simulates real-world decision-making for issuing credit based on historical customer behavior.

---

##  Repository Contents

- `Credit_Risk_Model.ipynb`: End-to-end notebook for data processing, feature engineering, modeling, and strategy simulation.
- `strategy_results.csv`: Final results for different thresholds — includes default rates and expected revenue.
- `revenue_vs_threshold.png`: Visualization of revenue across thresholds.

---

## Data Overview

Data source: [Amex Default Prediction – Kaggle](https://www.kaggle.com/competitions/amex-default-prediction/data)

- `train_labels.csv`: Default indicator as of April 2018.
- `train_data.csv`: 13-month history of customer transactions and credit activity.
- Sampled 20% of data to overcome memory constraints.

---

## Methodology

### Preprocessing

- One-hot encoding for categorical features.
- Aggregated temporal features (mean, max, recent value, etc.).
- Outlier treatment and normalization for Neural Networks.
- Missing values imputed with 0.

### Models Used

#### XGBoost
- Feature selection using importance threshold (> 0.5%).
- Grid search across:
  - Trees: 50, 100, 300
  - Learning Rate: 0.01, 0.1
  - Subsample & Colsample: 50%, 80%, 100%
  - Class weight: 1, 5, 10

####  Neural Network
- Built using Keras.
- Tuned on:
  - Hidden layers: 2, 4
  - Nodes: 4, 6
  - Activation: ReLU, Tanh
  - Dropout: 50%, None
  - Batch sizes: 100, 10,000

---

##  Lending Strategy

The final model output (Probability of Default) is used to implement two strategies:

- **Conservative**: Lower threshold → Low risk, low acceptance
- **Aggressive**: Higher threshold → Higher revenue, higher risk

A custom function calculates:

- **Portfolio Default Rate**
- **Expected Revenue** (based on historical `S_`pend and `B_`alance behavior)

---

## Results Summary

| Strategy     | Threshold | Default Rate | Expected Revenue | Accepted Customers |
|--------------|-----------|--------------|------------------|---------------------|
| Conservative | **0.17**  | **0.27%**     | **$902.70**      | 4,071               |
| Aggressive   | **0.58**  | **26.27%**    | **$1,719.64**    | 13,768              |

>  Full results are available in `strategy_results.csv`

---

##  Strategy Evaluation

The following chart visualizes how expected revenue varies with threshold selection:

![Revenue vs Threshold](output.png)



---

##  Tools Used

- Python, Pandas, NumPy
- XGBoost, Scikit-Learn, Keras/TensorFlow
- SHAP for explainability
- Matplotlib for visualization

---

##  Author

**Abhishek D Joy**  
M.S. Business Analytics | The University of Texas at Dallas  
axr230118@utdallas.edu
