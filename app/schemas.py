from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .data import clean_job_title

HomeOwnership = Literal['MORTGAGE', 'OTHER', 'OWN', 'RENT']
Verification = Literal['Not Verified', 'Source Verified', 'Verified']
Purpose = Literal['car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement',
                  'house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy',
                  'small_business', 'vacation', 'wedding']

# API field -> column name the model was trained on
COLUMNS = {
    'funded_amnt':           'funded_amnt',
    'interest_rate':         'interest_rate',
    'grade':                 'grade',
    'annual_income':         'annual_income',
    'dti':                   'dept_paym_income_ratio',
    'emp_length_years':      'emp_length_years',
    'home_ownership':        'home_ownership_status',
    'verification_status':   'verification_status',
    'loan_purpose':          'loan_purpose',
    'delinq_2yrs':           'num_30+_delinq_in_2yrs',
    'inquiries_6m':          'num_inq_in_6mths',
    'open_credit_lines':     'num_open_credit_lines',
    'revolving_balance':     'total_credit_revolving_bal',
    'revolving_utilization': 'used_credit_share',
    'credit_history_years':  'credit_history_years',
    'job_title':             'job_title'
}


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra='forbid')

    funded_amnt:           float = Field(ge=500, le=40000, description="Loan amount, USD")
    interest_rate:         float = Field(ge=5, le=31, description="Annual rate, %")
    grade:                 str = Field(pattern=r'^[A-G][1-5]$', description="Lending Club sub-grade, A1-G5")
    annual_income:         float = Field(gt=0, le=10_000_000)
    dti:                   float = Field(ge=0, le=60, description="Debt payments / income, %")
    emp_length_years:      Optional[int] = Field(default=None, ge=0, le=10, description="10 = 10+ years")
    home_ownership:        HomeOwnership
    verification_status:   Verification
    loan_purpose:          Purpose
    delinq_2yrs:           int = Field(ge=0, le=50, description="30+ day delinquencies, last 2 years")
    inquiries_6m:          int = Field(ge=0, le=40, description="Credit inquiries, last 6 months")
    open_credit_lines:     int = Field(ge=0, le=100)
    revolving_balance:     float = Field(ge=0, le=5_000_000)
    revolving_utilization: float = Field(ge=0, le=200, description="Revolving balance / limit, %")
    credit_history_years:  float = Field(ge=0, le=80, description="Years since first credit line")
    job_title:             Optional[str] = Field(default=None, max_length=100, description="Free text, e.g. Registered Nurse")

    def to_frame(self):
        row = {COLUMNS[k]: v for k, v in self.model_dump().items()}
        X = pd.DataFrame([row]).astype({'emp_length_years': float})
        X['job_title'] = clean_job_title(X['job_title'])
        return X


class Prediction(BaseModel):
    pd: float
    lgd: float
    ead: float
    expected_loss: float
    expected_loss_pct: float
    threshold: float
    decision: Literal['approve', 'reject']
