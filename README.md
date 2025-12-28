# Mercor Cheating Detection Kaggle Competition  
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)  

This repository contains my solution for the **Mercor Cheating Detection** Kaggle competition. The goal of this competition is to **predict whether a candidate is engaging in cheating behavior during online interviews**, using anonymized behavioral features, platform activity signals, and a social graph. The evaluation metric is **cost-sensitive**, reflecting the real-world operational impact of false negatives, false positives, and manual review decisions.

---

## 📝 Overview

- **Competition:** [Mercor Cheating Detection](https://www.kaggle.com/competitions/mercor-cheating-detection/overview)  
- **Task:** Binary classification (cheating vs. not cheating)  
- **Data:** Behavioral features (`feature_001`–`feature_018`), high-confidence clean flag, social graph of users  
- **Evaluation Metric:** Custom operational cost  
  - False Negative (missed cheating): \$600  
  - False Positive in manual review: \$150  
  - False Positive in auto-block: \$300  
  - True Positive requiring manual review: \$5  
  - Correct auto-pass or auto-block: \$0  

The submission requires predicting probabilities for each candidate and finding the **optimal decision thresholds** to minimize total operational cost.

---

## 📂 Repository Structure

├── Datasets/ # Raw and processed datasets (original datasets not included due to size)
| ├── Graph_train.csv # Processed train dataset with graph features
| ├── Graph_test.csv # Processed test dataset with graph features
| ├── train.csv # Original train dataset (download from Kaggle)
| ├── test.csv # Original test dataset (download from Kaggle)
| ├── referral_graph.csv # An edge list representing the complete social network (download from Kaggle)
├── Notebooks/ # Experiment notebooks
| ├── Mercor_Fraud_Add_Graph_Features.ipynb # Feature Engineering Notebook
| ├── Mercor_Fraud_Models_Graph_Features.ipynb # Model Building Notebook
├── README.md

---

## 🔧 Pipeline

### 1. Data Processing & Feature Engineering
- Loaded train, test, and referral graph datasets.  
- Engineered **graph-based features**:  
  - Node degree (`degree`)  
  - PageRank (`pagerank`)  
  - Connected component size (`component_size`)  
- Aggregated neighbor statistics from the social graph:  
  - Mean, standard deviation  
  - Ratios and differences between user and neighbors  
- Handled missing values using **Iterative Imputer** with `DecisionTreeRegressor`.  

### 2. Model Training & Stacking
- Used three base models:  
  - **LightGBM**  
  - **XGBoost**  
  - **CatBoost**  
- Trained with **5-fold Stratified K-Fold cross-validation**  
- Generated **out-of-fold (OOF) predictions** for meta-model input  
- Meta-model: **XGBoost** combining base model predictions  
- Final prediction: weighted ensemble of meta-model output and mean base predictions  

### 3. Threshold Optimization & Evaluation
- Defined three operational regions:  
  - Auto-pass  
  - Manual review  
  - Auto-block  
- Searched for optimal thresholds `(t1, t2)` to **minimize total operational cost**  
- Computed detailed counts: FN, TP (manual), FP (manual), FP (auto-block)  

### 4. Submission
- Final probabilities saved in `submission.csv` for Kaggle submission

---

## ⚙️ Key Libraries

pandas, numpy, networkx, scikit-learn, catboost, lightgbm, xgboost


---

## 💡 Highlights
- Semi-supervised learning with high-confidence unlabeled samples  
- Graph-based feature engineering to capture relational information  
- Cost-sensitive evaluation to mimic real operational impact  
- Multi-level stacking with optimized thresholding  

---

## 📈 Results
- **Leaderboard:** Ranked **160 / 358** participants in the Mercor Cheating Detection competition  
- **Best Score:** -1,587,335 (cost-based metric; lower is better)  
- Demonstrates ability to build **graph-based features**, handle **semi-supervised data**, and optimize **cost-sensitive thresholds**  
- Shows competency in **ensemble modeling, stacking, and practical ML pipelines**  

---

## 🔗 References
- [Mercor Cheating Detection Kaggle Competition](https://www.kaggle.com/competitions/mercor-cheating-detection/overview) – Official competition page.  
- Selected public Kaggle notebooks for inspiration on graph-based feature engineering and model stacking. **All code in this repository is implemented independently.**  

---

## 👨‍💻 How to Run

**Python version:** Tested with **Python 3.12.12** (used in Google Colab).  

**Datasets:** Original datasets are **not included** due to size. You can download them from the official Kaggle competition page:  
[Mercor Cheating Detection - Data](https://www.kaggle.com/competitions/mercor-cheating-detection/data)  

If running locally:  
- Place the downloaded dataset files (`train.csv`, `test.csv`, `referral_graph.csv`) in the `Datasets/` folder.  
- Update the paths in the notebooks if necessary.  
- Run the notebooks in order:  
  1. `Mercor_Fraud_Add_Graph_Features.ipynb` → generate `Graph_train.csv` and `Graph_test.csv`  
  2. `Mercor_Fraud_Models_Graph_Features.ipynb` → train models, optimize thresholds, and generate `submission.csv`  

