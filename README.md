# Credit Scoring & Expected Loss - Lending Club

![Python >= 3.13](https://img.shields.io/badge/Python-%3E%3D3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2E86AB?logo=lightgbm&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-Launch-brightgreen?logo=streamlit&logoColor=white)](https://credit-risk-pp.streamlit.app/)

This is a compehensive credit risk pipeline built on Lending Club loan data. Estimates the Expected Loss of a loan portfolio using the standard formula, with LGD (Loss Given Default) as a percentage:

> `EL = PD × (LGD / 100) × EAD`
> 
---
### SEE THE LIVE APP [HERE](https://credit-risk-pp.streamlit.app/)
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

In credit risk, missing a defaulter (FN) is far more costly than rejecting a good client (FP). In fact, it is worse to lose €100,000 from one client than $10,000 from 10 clients each
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
- Source (original): Lending Club Loan Data (Kaggle)
- Availability: The dataset is not included in this repository due to size.

## How to obtain the data
1. Download the dataset from Kaggle:
https://www.kaggle.com/datasets/db0boy/lending-club-loan-data-cleared/data
2. Place the original files in your working directory.
3. Run the EDA.ipynb file to generate:
```bash
cleaned_data.csv
```
4. Run main.py with the generated CSV.

## File Structure
```
.
├── Models/
│   └── metadata.json
├── Visualizations/
├── .gitignore
├── EDA.ipynb
├── LICENSE
├── main.py
├── modeling.py
├── README.md
├── requirements.txt
├── streamlit_app.py
├── utils.py
└── visualization.py
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook EDA.ipynb
python main.py
```

## Dependencies

pandas, numpy, matplotlib, seaborn, json, scikit-learn, lightgbm

## License

MIT License
