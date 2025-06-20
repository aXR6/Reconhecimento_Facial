# Reconhecimento Facial

Este projeto detecta rostos em imagens usando Python 3 e o modelo Haar Cascade do OpenCV.
Opcionalmente, é possível utilizar os modelos **MediaPipe-Face-Detection** ou **YOLOv8-Face-Detection** da Hugging Face localmente para auxiliar na detecção.
Os modelos são baixados automaticamente do Hub na primeira execução, não sendo necessário configurar URLs de API.
Também é possível gerar uma legenda da imagem utilizando um modelo de linguagem
da Hugging Face de forma local.

## Requisitos

- Python 3
- opencv-python (pode ser instalado com `pip install opencv-python`)
- mediapipe
- ultralytics
- torch
- transformers
- huggingface_hub
- python-dotenv

## Uso

1. Instale as dependências:
   ```
   pip install -r requirements.txt
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
   execute:
   ```
   python3 app.py
   ```
4. O script salva `saida.jpg` com retângulos ao redor dos rostos detectados.
