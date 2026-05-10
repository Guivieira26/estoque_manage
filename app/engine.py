from datetime import datetime
import easyocr
import requests
import json
import re
import logging
import os

logger = logging.getLogger(__name__)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
log_txt = "log.txt"

# AMD não usa CUDA; por padrão mantemos OCR em CPU.
# Se quiser forçar GPU em ambientes compatíveis, use EASYOCR_GPU=true.
USE_GPU = os.getenv("EASYOCR_GPU", "false").lower() in ("1", "true", "yes", "on")

# Inicializa o leitor em Português.
# Na primeira execução, ele baixará os modelos (~80MB).
reader = easyocr.Reader(['pt'], gpu=USE_GPU)
logger.info("EasyOCR iniciado em: %s", "GPU" if USE_GPU else "CPU")

def _normalizar_itens_ollama(dados):
    """Normaliza a resposta do Ollama para sempre retornar lista de dicts."""
    if isinstance(dados, list):
        return [item for item in dados if isinstance(item, dict)]

    if isinstance(dados, dict):
        # Caso comum: o modelo retorna um único item como objeto.
        if any(chave in dados for chave in ("nome", "name", "quantidade", "quantity", "preco", "price")):
            return [dados]

        # Caso alternativo: resposta embrulhada em uma chave conhecida.
        for chave in ("itens", "items", "produtos", "products", "data"):
            valor = dados.get(chave)
            if isinstance(valor, list):
                return [item for item in valor if isinstance(item, dict)]

    return []

def extrair_dados_com_ollama(texto_bruto):
    """
    Envia o texto bagunçado do OCR para o Phi-3 no Ollama 
    e retorna um JSON estruturado.
    """
    url = OLLAMA_URL
    
    prompt = f"""
    Analise este texto de uma nota fiscal (DANFE).
    Encontre a tabela de produtos e extraia: Nome, Quantidade e Valor Unitário.
    
    REGRAS:
    - Ignore cabeçalhos, impostos (ICMS/IPI) e dados do cliente.
    - Foque em linhas que tenham códigos numéricos seguidos de nomes de peças.
    - Se o nome for muito longo, resuma.
    - Responda EXCLUSIVAMENTE um JSON (Lista de Objetos).

    Exemplo de saída:
    [{{"nome": "PLACA MAE GIGABYTE", "quantidade": 1, "preco": 588.12}}]
    
    TEXTO OCR:
    {texto_bruto} 
    """
    # Note: Limitamos a 1500 caracteres para o Phi-3 não se perder no contexto.

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1 # Menos criatividade, mais precisão
        }
    }

    try:
        logger.info("Iniciando chamada ao Ollama para estruturar dados da nota. model=%s", OLLAMA_MODEL)
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # O Ollama retorna o JSON como uma string dentro do campo 'response'
        resultado_string = response.json().get('response', '[]')
        dados = json.loads(resultado_string)
        itens = _normalizar_itens_ollama(dados)
        logger.info("Ollama retornou %s item(ns) estruturado(s)", len(itens))

        # Log em arquivo txt
        with open(log_txt, "a", encoding="utf-8") as file:
            file.write(f"\n--- Processamento de Nota {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n") 
            file.write(f"Texto OCR:\n{texto_bruto}\n")
            file.write(f"Resposta Ollama:\n{json.dumps(itens, ensure_ascii=False, indent=2)}\n")


        return itens
    except Exception as e:
        logger.exception("Erro ao processar resposta do Ollama: %s", e)
        return []

def processar_imagem_nota(caminho_imagem):
    """
    Passo 1: OCR (Imagem -> Texto)
    Passo 2: LLM (Texto -> JSON)
    """
    # Extrai apenas o texto (detail=0 retorna uma lista de strings)
    logger.info("Iniciando OCR da imagem: %s", caminho_imagem)
    resultado_ocr = reader.readtext(caminho_imagem, detail=0)
    logger.info("OCR finalizado. Blocos de texto detectados: %s", len(resultado_ocr))
    texto_unificado = "\n".join(resultado_ocr)
    
    if not texto_unificado.strip():
        logger.warning("OCR não identificou texto útil na imagem: %s", caminho_imagem)
        return []

    # Envia para o filtro da LLM
    logger.info("Enviando texto do OCR para LLM. Tamanho do texto: %s caracteres", len(texto_unificado))
    dados_estruturados = extrair_dados_com_ollama(texto_unificado)
    logger.info("Processamento concluído. Itens estruturados: %s", len(dados_estruturados) if isinstance(dados_estruturados, list) else 0)
    return dados_estruturados