import json
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import PD_MODEL_PATH, LGD_MODEL_PATH, METRICS_PATH
from .schemas import LoanApplication, Prediction

warnings.filterwarnings('ignore', message='X does not have valid feature names')

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # models are loaded once, not per request
    state['pd_model']  = joblib.load(ROOT / PD_MODEL_PATH)
    state['lgd_model'] = joblib.load(ROOT / LGD_MODEL_PATH)
    with open(ROOT / METRICS_PATH) as f:
        state['metrics'] = json.load(f)
    yield
    state.clear()


app = FastAPI(title="Credit Risk Scoring", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/viz", StaticFiles(directory=ROOT / "viz"), name="viz")
templates = Jinja2Templates(directory=APP_DIR / "templates")

templates.env.filters['pct']   = lambda v, d=1: f"{v * 100:.{d}f}%"
templates.env.filters['num']   = lambda v, d=3: f"{round(v, d) + 0.0:.{d}f}"
templates.env.filters['money'] = lambda v: f"${v / 1e6:,.1f}M" if v >= 1e6 else f"${v:,.0f}"
templates.env.filters['int']   = lambda v: f"{int(v):,}"
templates.env.filters['period'] = lambda v: f"{str(v)[:4]}-{str(v)[4:]}"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"m": state['metrics']})


@app.post("/predict", response_model=Prediction)
def predict(loan: LoanApplication):
    X = loan.to_frame()
    policy = state['metrics']['policy']

    pd_ = float(state['pd_model'].predict_proba(X)[0, 1])
    lgd = float(np.clip(state['lgd_model'].predict(X)[0], 0, 1))
    ead = policy['ccf_train'] * loan.funded_amnt
    el  = pd_ * lgd * ead

    return Prediction(
        pd=round(pd_, 4),
        lgd=round(lgd, 4),
        ead=round(ead, 2),
        expected_loss=round(el, 2),
        expected_loss_pct=round(el / loan.funded_amnt, 4),
        threshold=policy['threshold'],
        decision='approve' if pd_ < policy['threshold'] else 'reject'
    )
