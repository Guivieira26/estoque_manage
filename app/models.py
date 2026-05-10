from pydantic import BaseModel

class ProdutoSchema(BaseModel):
    nome: str
    quantidade: int
    preco: float = 0.0