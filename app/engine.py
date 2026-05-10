from datetime import datetime
import pytesseract
from PIL import Image
import cv2
import numpy as np
import requests
import json
import re
import logging
import os

logger = logging.getLogger(__name__)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
log_txt = "log.txt"

TESSERACT_CONFIG = "--psm 6 -l por"
logger.info("Tesseract configurado. Config: %s", TESSERACT_CONFIG)

def extrair_texto_ocr(caminho_imagem: str) -> str:
    """
    Usa image_to_data para capturar posição de cada palavra.
    Agrupa palavras pela posição vertical (mesma linha = mesmo Y).
    Isso preserva a ordem correta das colunas da tabela DANFE,
    que o image_to_string quebra por ler coluna por coluna.
    """
    img_processada = preprocessar_imagem(caminho_imagem)
    pil_img = Image.fromarray(img_processada)

    # Extrai dados com posição de cada palavra
    dados = pytesseract.image_to_data(
        pil_img,
        config="--psm 1 -l por",  # psm 1 = detecção automática de layout
        output_type=pytesseract.Output.DICT
    )

    # Agrupa palavras por linha (top + altura define a faixa vertical)
    linhas = {}
    for i, palavra in enumerate(dados["text"]):
        palavra = palavra.strip()
        if not palavra or int(dados["conf"][i]) < 30:  # ignora baixa confiança
            continue

        # Usa o centro vertical da palavra como chave de agrupamento
        top = dados["top"][i]
        height = dados["height"][i]
        centro_y = top + height // 2

        # Tolerance de 8px: palavras no mesmo centro_y vão para a mesma linha
        chave_linha = (centro_y // 8) * 8

        if chave_linha not in linhas:
            linhas[chave_linha] = []
        linhas[chave_linha].append((dados["left"][i], palavra))

    # Ordena por Y (top->bottom) e dentro de cada linha por X (left->right)
    texto_final = []
    for y in sorted(linhas.keys()):
        palavras_na_linha = sorted(linhas[y], key=lambda p: p[0])
        linha_texto = " ".join(p[1] for p in palavras_na_linha)
        texto_final.append(linha_texto)

    resultado = "\n".join(texto_final)
    logger.info("OCR estruturado finalizado. Linhas reconstruídas: %s", len(texto_final))
    return resultado

#Pre processamento para o tesseract funcionar melhor
def preprocessar_imagem(caminho_imagem: str) -> np.ndarray:
    """
    Prepara a imagem para o Tesseract:
    - Converte para escala de cinza
    - Aumenta resolução se necessário (OCR melhora com DPI alto)
    - Remove ruído
    - Binariza (preto e branco) para separar texto do fundo
    """
    img = cv2.imread(caminho_imagem)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Se a imagem for pequena, aumenta para pelo menos 1500px de largura
    h, w = gray.shape
    if w < 1500:
        scale = 1500 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Suavização leve para remover ruído de câmera
    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    # Binarização adaptativa: lida bem com iluminação irregular em fotos
    binaria = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return binaria

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

#Evitar textos além de um JSON puro
def _parse_json_seguro(texto: str) -> list:
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    # Busca array JSON em qualquer parte da resposta
    match = re.search(r'\[.*\]', texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: objeto único virou array de 1
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        try:
            return [json.loads(match.group())]
        except json.JSONDecodeError:
            pass
    logger.warning("Não foi possível parsear JSON. Resposta bruta:\n%s", texto)
    return []

def extrair_dados_com_ollama(texto_bruto: str) -> list:
    """Envia o texto OCR completo para o LLM e retorna JSON estruturado."""

    prompt = f"""Você receberá o texto bruto de uma Nota Fiscal Eletrônica brasileira (DANFE).
Notas fiscais sempre têm uma tabela de produtos com estas colunas obrigatórias por lei:
NCM (8 dígitos), CFOP (4 dígitos começando com 5, 6 ou 7), quantidade e valor unitário.

TAREFA: Encontre cada produto identificando o padrão:
[descrição do produto] [NCM 8 dígitos] [CFOP 4 dígitos] [unidade] [quantidade] [valor unitário]

ATENÇÃO:
- Linhas sem NCM e CFOP são continuação da descrição do item anterior, não um novo produto.
- "TOT TRIB", "GARANTIA", "N.SERIE", "MESES", "IMEI" são informações fiscais, não produtos.
- O preço correto é o valor unitário, nunca o CFOP, NCM ou valor de tributo.
- Corrija erros de OCR: "oooo"=0, "Looog"=1, vírgula vira ponto em decimais.
- Extraia TODOS os produtos, do primeiro ao último. Nunca pare antes do fim.

TEXTO DA NOTA:
{texto_bruto}

Responda SOMENTE com o array JSON, sem texto antes ou depois:
[{{"nome": "...", "quantidade": 1, "preco": 0.00}}]

JSON:"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1024,
            "repeat_penalty": 1.1,
        }
    }

    try:
        logger.info("Chamando Ollama. model=%s", OLLAMA_MODEL)
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()

        resultado_string = response.json().get("response", "[]")
        dados = _parse_json_seguro(resultado_string)
        itens = _normalizar_itens_ollama(dados)
        logger.info("Ollama retornou %s item(ns)", len(itens))

        with open(log_txt, "a", encoding="utf-8") as file:
            file.write(f"\n--- Processamento de Nota {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            file.write(f"Modelo: {OLLAMA_MODEL}\n")
            file.write(f"Texto OCR:\n{texto_bruto}\n")
            file.write(f"Resposta Ollama:\n{json.dumps(itens, ensure_ascii=False, indent=2)}\n")

        return itens

    except requests.Timeout:
        logger.error("Timeout: Ollama demorou mais de 180s. Modelo pode estar sem RAM suficiente.")
        return []
    except requests.RequestException as e:
        logger.exception("Erro na requisição ao Ollama: %s", e)
        return []
    except Exception as e:
        logger.exception("Erro inesperado: %s", e)
        return []

def processar_imagem_nota(caminho_imagem):
    """
    Passo 1: OCR (Imagem -> Texto)
    Passo 2: LLM (Texto -> JSON)
    """
    # Extrai apenas o texto (detail=0 retorna uma lista de strings)
    logger.info("Iniciando OCR da imagem: %s", caminho_imagem)
    
    texto_unificado = extrair_texto_ocr(caminho_imagem)

    logger.info("OCR finalizado. Caracteres extraídos: %s", len(texto_unificado))
    
    if not texto_unificado.strip():
        logger.warning("OCR não identificou texto útil na imagem: %s", caminho_imagem)
        return []

    # Envia para o filtro da LLM
    logger.info("Enviando texto do OCR para LLM. Tamanho do texto: %s caracteres", len(texto_unificado))
    dados_estruturados = extrair_dados_com_ollama(texto_unificado)
    logger.info("Processamento concluído. Itens estruturados: %s", len(dados_estruturados) if isinstance(dados_estruturados, list) else 0)
    return dados_estruturados