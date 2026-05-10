Sistema de Estoque

- Automação de inclusão de produtos a partir de notas fiscais

> Tesseract para extração dos dados na foto;
> Modelos ollama para lidar com a String extraida e tratar para enviar como JSON

NOTE: " No momento qwen2.5 foi o melhor resultado até então "

to run command: ` uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload `
to acess venv -> source venv/bin/activate
sudo apt install tesseract-ocr tesseract-ocr-por
Modelos testados (mistral, qwen2.5:3b, gemma2:2b, phi3:latest)