# Credit Scoring & Expected Loss - Lending Club

![Python >= 3.13](https://img.shields.io/badge/Python-%3E%3D3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2E86AB?logo=lightgbm&logoColor=white)

This is an end-to-end credit risk pipeline built on Lending Club loan data. Estimates the Expected Loss of a loan portfolio using the standard formula, with LGD (Loss Given Default) as a percentage:

$$
EL = PD \times (LGD/100) \times EAD
$$

## Project Overview

The project is structured in 4 phases:

- Data Preparation: The raw Lending Club dataset comes with noise, nulls and post-default variables that would cause data leakage to the target variable. We impute those, delete some irrelevant variables, cap outliers and encode cardinal variables before any model sees the data.
- PD Model: Binary classification to estimate the probability of not paying for each loan.
- LGD Model: Regression trained exclusively on defaulted loans to predict loss magnitude.
- Expected Loss: Loss estimation combining both models into a single dollar figure, applying the formula above.

## Estimated Models

### PD: Classification (ROC-AUC)

| Model               | ROC-AUC        |
| ------------------- | -------------- |
| **LightGBM**  | **0.75** |
| Logistic Regression | 0.74           |
| Random Forest       | 0.73           |
| Decision Tree       | 0.55           |

> LightGBM is the best model here. Appart from having the highest ROC-AUC, it is also the fastest and most complete model. I have set the threshold to 0.10 to maximize Recall, because in credit risk, missing a defaulter is more costly, in fact, it is worse to lose €100,000 from one client than $10,000 from 10 clients each.

### LGD: Regression (MAE & RMSE)

| Model                   | MAE            | RMSE           |
| ----------------------- | -------------- | -------------- |
| Random Forest           | 0.25           | 1.38           |
| LightGBM                | 0.41           | 1.48           |
| Decision Tree           | 0.56           | 2.65           |
| Linear Regression       | 11.39          | 20.09          |

> Random Forest selected as best regression model with MAE of 0.25%.

## Expected Loss Result

| Metric               | Value        |
| -------------------- | ------------ |
| Portfolio (test set) | $174,008,000 |
| Expected Loss        | $7,815,756   |
| % of portfolio       | 4.49%        |

> A 4.49% expected loss is within acceptable range for consumer credit, offset by Lending Club's interest rates. Recommended to provision this amount prior to portfolio acquisition.

## Dataset
- Source (original): Lending Club Loan Data (Kaggle)
- Availability: The dataset is not included in this repository due to size.

## How to obtain the data
1. Download the dataset from Kaggle:
https://www.kaggle.com/datasets/db0boy/lending-club-loan-data-cleared/data
2. Place the original files in your working directory.
3. The dataset already includes: Run the preprocessing pipeline to generate:
```bash
cleaned_data.csv
X.csv
target.csv
```

## File Structure
```
.
├── Models/
│   ├── best_clf_model.pkl
│   └── best_reg_model.pkl
├── Visualizations/
├── .gitignore
├── EDA.ipynb
├── LICENSE
├── main.py
├── modeling.py
├── README.md
├── requirements.txt
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

pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm, joblib

## License

MIT License
