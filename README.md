# Credit Scoring & Expected Loss - Lending Club

![Python >= 3.13](https://img.shields.io/badge/Python-%3E%3D3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2E86AB?logo=lightgbm&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-Launch-brightgreen?logo=streamlit&logoColor=white)](https://credit-risk-pp.streamlit.app/)
![License](https://img.shields.io/badge/License-MIT-green)

This is a comprehensive credit risk pipeline built on Lending Club loan data. Estimates the Expected Loss of a loan portfolio using the standard formula, with LGD (Loss Given Default) as a percentage:

> `EL = PD × (LGD / 100) × EAD`
> 
---
### 👉 [Live Demo Here](https://credit-risk-pp.streamlit.app/)
---

## Project Overview

The project is structured in 4 phases:

- Data Preparation: The raw Lending Club dataset comes with noise, nulls and post-default variables that would cause data leakage to the target variable. We impute those, delete some irrelevant variables, cap outliers and encode cardinal variables before any model sees the data.
- PD Model: Binary classification to estimate the probability of not paying for each loan.
- LGD Model: Regression trained exclusively on defaulted loans to predict loss magnitude.
- Expected Loss: Loss estimation combining both models into a single dollar figure, applying the formula above.

## Estimated Models

### PD: Classification (Focus on maximizing ROC-AUC)

| Model               | ROC-AUC        |
| ------------------- | -------------- |
| **LightGBM**  | **0.75** |
| Logistic Regression | 0.74           |
| Random Forest       | 0.73           |
| Decision Tree       | 0.55           |

> LightGBM is the best model here. Appart from having the highest ROC-AUC, it is also the fastest and most complete model.

### Business Decision: Threshold Selection

The threshold is set at 0.10 to prioritize risk reduction.

- Recall = 0.95 → 95% of defaulters are identified  
- Precision = 0.54 → moderate false positives  

In credit risk, missing a defaulter (FN) is far more costly than rejecting a good client (FP). In fact, it is worse to lose $100,000 from one client than $10,000 from 10 clients each
This threshold minimizes costly defaults, aligning with a **conservative risk strategy**.

> Precision-Recall Trade-off: lower approvals, higher portfolio quality.

### LGD: Regression (Focus on minimizing MAE & RMSE)

| Model                   | MAE            | RMSE           |
| ----------------------- | -------------- | -------------- |
| Random Forest           | 0.25           | 1.38           |
| LightGBM                | 0.41           | 1.48           |
| Decision Tree           | 0.56           | 2.65           |
| Linear Regression       | 11.39          | 20.09          |

> Random Forest is the best regression model here, with a MAE of 0.25%.

## Expected Loss Result

| Metric               | Value        |
| -------------------- | ------------ |
| Portfolio (test set) | $174,008,000 |
| Expected Loss        | $7,815,756   |
| % of portfolio       | 4.49%        |

> 4.49% expected loss. Acceptable range for consumer credit.

## Dataset

The dataset is not included due to size constraints.

Download it from Kaggle and preprocess it using ETL.ipynb.

## File Structure
```
.
├── app/
│   ├── __init__.py
│   ├── main.py                  # model training
│   ├── modeling.py
│   ├── utils.py
│   └── visualization.py
├── Models/
│   └── metadata.json
├── Visualizations/
├── .gitignore
├── ETL.ipynb                   # Data quality, feature engineering & risk analysis
├── LICENSE
├── README.md
├── requirements.txt
└── streamlit_app.py            # deployed application
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run ETL (Data Preparation)
Open and execute the Jupyter notebook to generate cleaned data:
```
ETL.ipynb
```
This generates `Data/cleaned_data.csv`

### 3. Train Models
```bash
python -m app.main
```
This trains PD (classification) and LGD (regression) models and generates `Models/metadata.json`

### 4. Launch Dashboard
```bash
streamlit run streamlit_app.py
```

> **Note:** For production, the app is already deployed at [credit-risk-pp.streamlit.app](https://credit-risk-pp.streamlit.app/)

## License

MIT License
