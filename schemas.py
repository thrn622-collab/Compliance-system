
# الملف: schemas.py

from pydantic import BaseModel
from typing import Optional, List

class ClientCreate(BaseModel):
    client_type: str # 'Individual' or 'Corporate'
    full_name: str
    identifier_number: str
    nationality: str
    date_of_birth_or_inc: str
    mobile_number: str
    activity_sector: str
    financial_value: Optional[float] = 0.0

class ClientResponse(ClientCreate):
    id: int
    risk_level: str

    class Config:
        from_attributes = True

class TransactionReportResponse(BaseModel):
    id: int
    client_id: int
    file_name: str
    total_transactions: int
    suspicious_count: int
    ai_decision: str
    
    class Config:
        from_attributes = True
