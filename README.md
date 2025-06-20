# Reconhecimento Facial

Este projeto detecta rostos em imagens usando Python 3 e o modelo Haar Cascade do OpenCV.
Opcionalmente, é possível utilizar o modelo **MediaPipe-Face-Detection** da Hugging Face para auxiliar na detecção.
Também é possível gerar uma legenda da imagem utilizando um modelo de linguagem
da Hugging Face via API.

## Requisitos

- Python 3
- opencv-python (pode ser instalado com `pip install opencv-python`)
- requests
- huggingface_hub

## Uso

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
2. Você pode executar a detecção diretamente:
   ```
   python3 face_detection.py --image caminho/para/imagem.jpg --output saida.jpg
   ```
   Para utilizar o modelo da Hugging Face adicione a opção `--hf`:
   ```
   python3 face_detection.py --image caminho/para/imagem.jpg --hf
   ```
3. Para uma experiência interativa com menu e opção de gerar legendas via LLM,
   execute:
   ```
   python3 app.py
   ```
   Para utilizar a geração de legenda é necessário definir a variável de
   ambiente `HUGGINGFACE_TOKEN` com um token válido da Hugging Face.
4. O script salva `saida.jpg` com retângulos ao redor dos rostos detectados.
