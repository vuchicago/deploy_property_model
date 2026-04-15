from database import Base
from sqlalchemy import Column, Integer, String, Boolean, Float


class TransactionProperty(Base):
    __tablename__ = 'transactions_all' #create sql lite table

    id = Column(Integer, primary_key=True, index=True)
    Description = Column(String)
    Prediction = Column(String)
    Proba= Column(Float)

