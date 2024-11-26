from pydantic import BaseModel

    
class Document(BaseModel):
    base64_file: str
    