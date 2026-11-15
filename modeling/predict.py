#!/usr/bin/python3
import pickle

import uvicorn

from pathlib import Path
from fastapi import FastAPI
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Election(BaseModel):
    """Validation for a given election input to model"""

    # Election filing columns
    case_type: Literal['RM', 'RC']
    state: Literal['IN', 'OH', 'CA', 'MS', 'OR', 'CO', 'ME', 'WV', 'PA', 'IA',
        'DC', 'TN', 'MI', 'NH', 'NY', 'MN', 'GA', 'TX', 'IL', 'OK', 'LA', 'NJ', 
        'ID', 'CT', 'MD', 'MO', 'KY', 'WA', 'FL', 'AR', 'MA', 'SC', 'KS', 'AZ',
        'GU', 'NV', 'VA', 'HI', 'NM', 'WI', 'NC', 'SD', 'RI', 'NE', 'UT', 'ND', 
        'VI', 'MT', 'DE', 'AL', 'PR', 'VT', 'WY', 'AK']
    unit_size: int = Field(..., ge=0)
    turnout: float = Field(..., ge=0.0, le=1.0)
    days_to_election: int = Field(..., ge=0)

    # Union finance columns
    log_members: int = Field(..., ge=-20, le=20)
    log_rep_per_mem: int = Field(..., ge=-20, le=20)
    log_admin_per_mem: int = Field(..., ge=-20, le=20)
    finance_aff_abbr: Literal['USW', 'IAM', 'UE', 'ATU','IATSE', 'NEA',
        'LIUNA', 'IBT', 'UFCW', 'GUA','UAW', 'OPEIU', 'IBEW', 'IUOE', 
        'BCTGMI', 'SEIU','ILWU', 'PPF', 'HFIA', 'RWDSU', 'ANA', 'UNITHE',
        'UNAFF', 'SMART', 'IUEC', 'PAT', 'AFM', 'WU','USWU', 'AFSCME',
        'PTE', 'CWA', 'TWU', 'UWU','APWU', 'SPFPA', 'UGSOA', 'AFT', 
        'OPCM', 'CJA','NNU', 'AFGE', 'BSOIW', 'NPWU', 'NAGE', 'IUJAT',
        'UAPD', 'WPPW', 'CLA', 'MEBA', 'IAFF', 'ILA','BAC', 'PPPWU',
        'RWAW', 'NOITU', 'UWA', 'AFA','NPW', 'SIUNA', 'MTD', 'BLE', 
        'NATCA', 'BBF','UNAP', 'UMW', 'BCTD', 'BRS', 'UTU', 'AAA', 
        'ACOPS', 'NACU', 'IWW', 'PPAN', 'SAGAFTRA', 'LEOS-PBA','RLCA', 
        'NSOI', 'NFFE', 'GMP', 'BMWE', None]


class PredictedOutcome(BaseModel):
    """Output of prediction"""
    success_prob: float
    election_success: bool


app = FastAPI(title="model_predictions")

# Path for local structure
model_file = PROJECT_ROOT / "data" / "random_forest_model.bin"
# Path when deploying with Docker
model_file = "data/random_forest_model.bin"
with open(model_file, "rb") as f_in:
    pipeline = pickle.load(f_in)


@app.post("/predict")
def predict(election: Election) -> PredictedOutcome:
    """"""
    election_dict = election.model_dump()

    threshold = 0.5
    success_prob = pipeline.predict_proba(election_dict)[0, 1]

    return PredictedOutcome(
        success_prob=success_prob,
        election_success=(success_prob > threshold)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)