import shutil
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from .database import init_db, get_db_connection, get_db_path
from .engine import processar_imagem_nota
from .models import ProdutoSchema
import os
import datetime
import logging
from pathlib import Path



app = FastAPI(title="Estoque - API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
else:
    logging.getLogger().setLevel(logging.INFO)

# Use absolute path for templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Inicializa o banco de dados ao ligar o servidor
@app.on_event("startup")
def startup_event():
    init_db()
    # Log do caminho do arquivo SQLite usado pela aplicação
    logger.info("Banco SQLite usado: %s", get_db_path())

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    # Isso renderiza o seu index.html na página inicial
    return templates.TemplateResponse("index.html", {"request": request})

# Rota para o celular listar o estoque
@app.get("/produtos")
def listar_produtos():
    conn = get_db_connection()
    produtos = conn.execute('SELECT * FROM products ORDER BY id DESC').fetchall()
    conn.close()
    itens = [dict(p) for p in produtos]
    logger.info("Listando produtos (%s itens) - DB: %s", len(itens), get_db_path())
    return itens

# Rota para o celular adicionar um produto em estoque
def adicionar_ou_atualizar_produto(produto: ProdutoSchema):
    now = datetime.datetime.now().isoformat()
    conn = get_db_connection()
    conn.execute('''
                INSERT INTO products (name, quantity, price, date_added) VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET quantity = quantity + excluded.quantity, price = excluded.price, date_added = excluded.date_added
                 ''',
        (produto.nome.upper(), produto.quantidade, produto.preco, now)
    )
    conn.commit()
    conn.close()


@app.post("/produtos")
def adicionar_produto(produto: ProdutoSchema):
    try:
        adicionar_ou_atualizar_produto(produto)
        return {"status": "Sucesso","item": produto.nome}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#rota de venda
@app.post("/venda")
def registrar_venda(nome: str, qtd_venda: int):
    conn = get_db_connection()
    # Busca o produto para ver se tem estoque
    produto = conn.execute('SELECT * FROM products WHERE name = ?', (nome.upper(),)).fetchone()
    
    if not produto:
        conn.close()
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    
    if produto['quantity'] < qtd_venda:
        conn.close()
        raise HTTPException(status_code=400, detail="Estoque insuficiente")
    
    # Subtrai a quantidade
    conn.execute('UPDATE products SET quantity = quantity - ? WHERE name = ?', (qtd_venda, nome.upper()))
    conn.commit()
    conn.close()
    return {"status": "venda realizada", "restante": produto['quantity'] - qtd_venda}

#Rota upload nota fiscal
# Pasta para salvar as fotos temporariamente
UPLOAD_DIR = str(Path(__file__).parent.parent / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-nota")
async def upload_nota(file: UploadFile = File(...)):
    caminho_arquivo = os.path.join(UPLOAD_DIR, file.filename)
    logger.info("Upload recebido: filename=%s content_type=%s", file.filename, file.content_type)
    
    # 1. Salva a imagem no disco
    with open(caminho_arquivo, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.info("Arquivo salvo em disco: %s", caminho_arquivo)
    
    # 2. Processa a imagem (OCR + LLM)
    # Nota: Isso pode demorar alguns segundos na CPU
    itens_identificados = processar_imagem_nota(caminho_arquivo)
    logger.info("Itens identificados pelo engine: %s", len(itens_identificados))
    
    # 3. Salva os itens no Banco de Dados automaticamente
    sucessos = []
    for item in itens_identificados:
        if not isinstance(item, dict):
            logger.warning("Item ignorado por formato inválido: %s", item)
            continue

        try:
            nome = str(item.get('nome') or item.get('name') or 'DESCONHECIDO').strip().upper()
            quantidade = int(item.get('quantidade', item.get('quantity', 0)) or 0)
            preco = float(item.get('preco', item.get('price', 0.0)) or 0.0)

            # Reutiliza a lógica de UPSERT da Sprint 1
            novo_prod = ProdutoSchema(
                nome=nome,
                quantidade=quantidade,
                preco=preco
            )
            # Aqui você chama sua função de adicionar ao banco criada na Sprint 1
            adicionar_ou_atualizar_produto(novo_prod)
            sucessos.append(novo_prod.nome)
        except Exception as e:
            logger.exception("Falha ao salvar item identificado: %s | erro=%s", item, e)
            continue

    logger.info("Upload finalizado. Itens persistidos: %s", len(sucessos))

    return {
        "status": "processado", 
        "itens_adicionados": sucessos,
        "total_identificado": len(itens_identificados)
    }