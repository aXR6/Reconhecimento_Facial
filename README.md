# Reconhecimento Facial

Este projeto detecta rostos em imagens usando Python 3 e o modelo Haar Cascade do OpenCV.
Opcionalmente, é possível utilizar os modelos **MediaPipe-Face-Detection**, **YOLOv8-Face-Detection** ou **Face-Obstruction-Detection** da Hugging Face localmente para auxiliar na detecção.
Os modelos são baixados automaticamente do Hub na primeira execução, não sendo necessário configurar URLs de API.
Também é possível gerar uma legenda da imagem utilizando um modelo de linguagem da Hugging Face de forma local.

O projeto inclui uma interface de linha de comando unificada e testes automatizados com `pytest`. Modelos alternativos podem ser definidos pelas variáveis de ambiente `MEDIAPIPE_REPO`, `YOLOV8_REPO`, `HF_CAPTION_MODEL` e `OBSTRUCTION_MODEL_REPO`.
Todas essas variáveis podem ser configuradas em um arquivo `.env` na raiz do projeto e serão carregadas automaticamente.

Agora a aplicação também oferece:

- Processamento de vídeos ou webcam com `--video` ou `--camera`.
- Reconhecimento facial com cadastro de pessoas.
- Cadastro de pessoas via webcam.
- Reconhecimento em tempo real pela webcam.
- Classificação de emoções.
- Opção de desfocar rostos para privacidade.
- Resultados gravados em banco PostgreSQL (configurável pela variável `POSTGRES_DSN`).
- Interface web em Flask (`web_app.py`).
- Dockerfile para facilitar a execução.
- Menu interativo mais elegante usando `questionary`.

Copie o arquivo `.env.example` para `.env` e ajuste conforme necessário. Todas as dependências podem ser instaladas utilizando o `pyproject.toml`.

## Variáveis de ambiente

- `MEDIAPIPE_REPO`: repositório do modelo MediaPipe (padrão: `qualcomm/MediaPipe-Face-Detection`).
- `YOLOV8_REPO`: repositório do modelo YOLOv8 (padrão: `jaredthejelly/yolov8s-face-detection`).
- `HF_CAPTION_MODEL`: modelo de legenda (padrão: `nlpconnect/vit-gpt2-image-captioning`).
- `OBSTRUCTION_MODEL_REPO`: modelo para detectar obstrução facial
  (padrão: `dima806/face_obstruction_image_detection`).
- `EMOTION_MODEL_REPO`: modelo de classificação de emoções (padrão: `nateraw/fer-vit-base`).
- `POSTGRES_DSN`: string de conexão do PostgreSQL usada por `db.py`.

## Requisitos

- Python 3
- opencv-python (pode ser instalado com `pip install opencv-python`)
- mediapipe
- ultralytics
- torch >=2.6 (necessário devido às proteções do `transformers` contra CVE-2025-32434)
- transformers
- huggingface_hub
- python-dotenv

## Uso

1. Instale as dependências:
   ```
   pip install .
   ```
   Em seguida crie as tabelas necessárias (PostgreSQL deve estar em
   execução e a variável `POSTGRES_DSN` configurada):
   ```
   python -c "import db; db.init_db()"
   ```
2. Você pode executar a detecção diretamente:
   ```
   python3 face_detection.py --image caminho/para/imagem.jpg --output saida.jpg
   ```
   Para utilizar um dos modelos da Hugging Face adicione a opção `--hf`. O
   modelo padrão é o `mediapipe`, mas é possível usar o YOLOv8 passando
   `--model yolov8`:
   ```
   python3 face_detection.py --image caminho/para/imagem.jpg --hf --model yolov8
   ```
3. Para uma experiência interativa com menu e opção de gerar legendas via LLM,
   ou para executar comandos diretamente, utilize:
   ```
   python3 app.py           # menu interativo
    python3 app.py detect --image caminho/para/imagem.jpg --model yolov8
    python3 app.py caption --image caminho/para/imagem.jpg
    python3 app.py obstruction --image caminho/para/imagem.jpg
   ```
4. O script salva `saida.jpg` com retângulos ao redor dos rostos detectados.
5. Para cadastrar uma pessoa usando a webcam, escolha a opção "Cadastrar pessoa" no menu interativo. A imagem da webcam fica aberta até pressionar **c** ou **espaço** para capturar (ou **q** para cancelar).
6. Para reconhecimento facial em tempo real pela webcam utilize a opção do menu
   interativo ou execute diretamente:
   ```
   python3 recognition.py --webcam
   ```

## Testes

Execute:
```bash
pytest
```

## Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para detalhes.
