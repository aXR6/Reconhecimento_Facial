# Reconhecimento Facial

Este projeto utiliza Python e modelos locais para detectar rostos em imagens ou na webcam. Também é possível gerar legendas descritivas das fotos e realizar reconhecimento facial em tempo real.

## Clonando o projeto

```bash
git clone https://github.com/seu-usuario/Reconhecimento_Facial.git
cd Reconhecimento_Facial
```

## Ambiente virtual

No Linux você pode iniciar o ambiente de desenvolvimento com o script `init-env.sh` que acompanha o repositório:

```bash
source init-env.sh
```

Esse script cria (caso ainda não exista) e ativa o virtualenv automaticamente.

## Executando

Instale as dependências e crie as tabelas necessárias (caso use PostgreSQL):

```bash
pip install .
python -c "from reconhecimento_facial import db; db.init_db()"
```

Para utilizar o menu interativo:

```bash
python3 -m reconhecimento_facial.app
```

Dentro de **Outros**, utilize a op\u00e7\u00e3o *Selecionar backend demogr\u00e1fico* para alternar entre FaceXFormer e DeepFace.

Você também pode executar comandos específicos diretamente pela CLI:

```bash
python3 -m reconhecimento_facial.face_detection --image caminho/para/imagem.jpg --output saida.jpg
python3 -m reconhecimento_facial.app detect --image caminho/para/imagem.jpg --model yolov8
python3 -m reconhecimento_facial.recognition --webcam
```

## Organização dos menus

O programa principal (`app.py`) apresenta três categorias principais:

1. **Detecção** – para identificar rostos ou possíveis obstruções nas imagens.
2. **Reconhecimento** – para cadastrar pessoas e fazer reconhecimento facial via webcam.
3. **Outros** – onde é possível gerar legendas para imagens usando um modelo de linguagem e escolher a biblioteca de análise demográfica.

## Funcionalidades

- Processamento de vídeos ou webcam.
- Cadastro de pessoas e reconhecimento em tempo real.
- Detecção de obstrução facial.
- Detecção de sexo, idade, etnia e cor de pele na webcam.
- Escolha entre FaceXFormer ou DeepFace para análise demográfica.
- Opção de desfocar rostos para privacidade.
- Armazenamento de resultados em PostgreSQL (via `POSTGRES_DSN`).
- Interface web em Flask e Dockerfile para facilitar a execução.

Copie o arquivo `.env.example` para `.env` e ajuste conforme necessário. Todas as dependências podem ser instaladas utilizando o `pyproject.toml`.

## Variáveis de ambiente

- `MEDIAPIPE_REPO`: repositório do modelo MediaPipe (padrão: `qualcomm/MediaPipe-Face-Detection`).
- `YOLOV8_REPO`: repositório do modelo YOLOv8 (padrão: `jaredthejelly/yolov8s-face-detection`).
- `HF_CAPTION_MODEL`: modelo de legenda (padrão: `nlpconnect/vit-gpt2-image-captioning`).
- `OBSTRUCTION_MODEL_REPO`: modelo para detectar obstrução facial (padrão: `dima806/face_obstruction_image_detection`).
- `FACEXFORMER_REPO`: repositório do FaceXFormer utilizado para estimar sexo, idade, etnia e cor de pele.
- `DEMOGRAPHICS_BACKEND`: biblioteca usada para estimar sexo e idade (`facexformer` ou `deepface`).
- `POSTGRES_DSN`: string de conexão do PostgreSQL usada por `db.py`.

## Requisitos

- Python 3
- opencv-python
- mediapipe
- ultralytics
- torch >=2.6
- transformers
- huggingface_hub
- python-dotenv
- deepface

## Testes

Execute os testes automatizados com:

```bash
pytest
```

## Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para detalhes.


