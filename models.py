# الملف: backend/models.py

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from db import Base

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    client_type = Column(String) # 'Individual' or 'Corporate'
    full_name = Column(String, index=True)
    identifier_number = Column(String, unique=True, index=True) 
    nationality = Column(String)
    date_of_birth_or_inc = Column(String)
    mobile_number = Column(String)
    activity_sector = Column(String)
    financial_value = Column(Float, nullable=True)
    risk_level = Column(String, default="Low") 
    
    # علاقات الربط
    documents = relationship("Document", back_populates="owner")
    reports = relationship("TransactionReport", back_populates="owner")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    document_type = Column(String)
    file_path = Column(String)
    
    owner = relationship("Client", back_populates="documents")

class TransactionReport(Base):
    __tablename__ = "transaction_reports"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id")) 
    file_name = Column(String) 
    total_transactions = Column(Integer) 
    suspicious_count = Column(Integer) 
    ai_decision = Column(String) 
    
    owner = relationship("Client", back_populates="reports")