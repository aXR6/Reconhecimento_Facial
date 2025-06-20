# Reconhecimento Facial

Este projeto detecta rostos em imagens usando Python 3 e o modelo Haar Cascade do OpenCV.

## Requisitos

- Python 3
- opencv-python (pode ser instalado com `pip install opencv-python`)

## Uso

1. Instale as dependências:
   ```
   pip install opencv-python
   ```
2. Execute a detecção em uma imagem:
   ```
   python3 face_detection.py --image caminho/para/imagem.jpg --output saida.jpg
   ```
3. O script salva `saida.jpg` com retângulos ao redor dos rostos detectados.

Um exemplo pode ser feito com a imagem `sample.jpg` incluída no repositório:
```bash
python3 face_detection.py --image sample.jpg --output result.jpg
```
O resultado será salvo em `result.jpg`.
