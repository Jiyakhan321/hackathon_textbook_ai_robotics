from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    selected_text: Optional[str] = None

class Source(BaseModel):
    title: str
    source: str

class QueryResponse(BaseModel):
    response: str
    sources: List[Source]
    query: str