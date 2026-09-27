import pytest
from fastapi.testclient import TestClient
from typing import get_args

from app.main import app, state
from app.schemas import HomeOwnership, Verification, Purpose

LOAN = {
    'funded_amnt': 12000,
    'interest_rate': 12.5,
    'grade': 'B4',
    'annual_income': 65000,
    'dti': 17,
    'emp_length_years': 6,
    'home_ownership': 'MORTGAGE',
    'verification_status': 'Source Verified',
    'loan_purpose': 'debt_consolidation',
    'delinq_2yrs': 0,
    'inquiries_6m': 0,
    'open_credit_lines': 11,
    'revolving_balance': 11000,
    'revolving_utilization': 50,
    'credit_history_years': 15,
    'job_title': 'Registered Nurse'
}

RISKY = {**LOAN, 'grade': 'G5', 'interest_rate': 29.9, 'dti': 45, 'inquiries_6m': 6,
         'delinq_2yrs': 3, 'revolving_utilization': 110, 'emp_length_years': None}


@pytest.fixture(scope='module')
def client():
    # the context manager runs the lifespan, which loads the models
    with TestClient(app) as c:
        yield c


def test_index(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'Probability of default' in r.text


def test_predict_valid(client):
    r = client.post('/predict', json=LOAN)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {'pd', 'lgd', 'ead', 'expected_loss', 'expected_loss_pct', 'threshold', 'decision'}
    assert body['decision'] in ('approve', 'reject')
    assert body['expected_loss'] == pytest.approx(body['pd'] * body['lgd'] * body['ead'], rel=1e-2)


@pytest.mark.parametrize('loan', [LOAN, RISKY], ids=['typical', 'risky'])
def test_pd_and_lgd_in_unit_interval(client, loan):
    body = client.post('/predict', json=loan).json()
    assert 0 <= body['pd'] <= 1
    assert 0 <= body['lgd'] <= 1


def test_riskier_loan_gets_higher_pd(client):
    low = client.post('/predict', json=LOAN).json()['pd']
    high = client.post('/predict', json=RISKY).json()['pd']
    assert high > low


@pytest.mark.parametrize('change', [
    {'grade': 'Z9'},
    {'funded_amnt': -100},
    {'interest_rate': 'high'},
    {'home_ownership': 'CASTLE'},
    {'unexpected_field': 1}
], ids=['bad_grade', 'negative_amount', 'non_numeric', 'unknown_category', 'extra_field'])
def test_predict_invalid(client, change):
    r = client.post('/predict', json={**LOAN, **change})
    assert r.status_code == 422


@pytest.mark.parametrize('title', [None, '', 'Chief Dragon Tamer'], ids=['none', 'empty', 'unseen'])
def test_job_title_optional_and_unseen(client, title):
    r = client.post('/predict', json={**LOAN, 'job_title': title})
    assert r.status_code == 200
    assert 0 <= r.json()['pd'] <= 1


def test_predict_missing_field(client):
    loan = {k: v for k, v in LOAN.items() if k != 'annual_income'}
    assert client.post('/predict', json=loan).status_code == 422


def test_schema_categories_seen_in_training(client):
    # every category the API accepts must exist in the training data
    seen = state['metrics']['features']['categories']
    assert set(get_args(HomeOwnership)) <= set(seen['home_ownership_status'])
    assert set(get_args(Verification)) <= set(seen['verification_status'])
    assert set(get_args(Purpose)) <= set(seen['loan_purpose'])
