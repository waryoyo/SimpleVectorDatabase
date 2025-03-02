from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text
from database import Base


class IndexMetadata(Base):
    __tablename__ = "IndexMetadata"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    model_name = Column(String)
    extra_metadata = Column(Text, nullable=True)


class TextInput(BaseModel):
    text: str
    metadata: Optional[Dict] = Field(default={})


class AddVectorInput(BaseModel):
    documents: List[TextInput]


class QueryInput(BaseModel):
    query: str
    k: int = 3
