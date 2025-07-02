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
Execute `face-preload` para baixar todos os modelos necessarios antes do primeiro uso.

## Executando

Instale as dependências e crie as tabelas necessárias (caso use PostgreSQL):

```bash
pip install .
python -c "from reconhecimento_facial import db; db.init_db()"
```
O script `init_db()` cria as tabelas `detections` e `people` de acordo com
`Estrutura_SQL/schema.sql`. A tabela `people` inclui a coluna `photo` para
armazenar a foto utilizada no cadastro de cada pessoa.

Para utilizar o menu interativo:

```bash
python3 -m reconhecimento_facial.app
```

Os modelos necessários são baixados automaticamente na primeira vez em que cada funcionalidade é utilizada.

Na seção **Outros** é possível escolher o dispositivo de processamento (CPU ou GPU).

Você também pode executar comandos específicos diretamente pela CLI:

```bash
python3 -m reconhecimento_facial.face_detection --image caminho/para/imagem.jpg --output saida.jpg
python3 -m reconhecimento_facial.app detect --image caminho/para/imagem.jpg --model yolov8
python3 -m reconhecimento_facial.recognition --webcam
python3 -m reconhecimento_facial.whisper_translation --model openai/whisper-large-v3-turbo --chunk 5 --src pt --webcam
python3 -m reconhecimento_facial.whisper_translation --file caminho/para/audio.wav --src pt --expected "texto esperado"
python3 -m reconhecimento_facial.whisper_translation --file caminho/para/audio.wav --transcribe --src pt
```

## Organização dos menus

O programa principal (`app.py`) apresenta quatro categorias principais:

1. **Detecção** – identifica rostos (inclusive via webcam) ou obstruções nas imagens.
2. **Reconhecimento** – para realizar reconhecimento facial ou exibir demografia via webcam.
3. **Cadastrar pessoa** – registra uma nova pessoa usando a webcam ou uma imagem com `--register-image`
4. **Outros** – onde é possível gerar legendas para imagens usando um modelo de linguagem.

## Funcionalidades

- Processamento de vídeos ou webcam.
- Cadastro de pessoas e reconhecimento em tempo real.
- Detecção de obstrução facial.
- Detecção de sexo, idade, etnia e cor de pele na webcam.
- Análise facial completa via FaceXFormer (segmentação, landmarks, pose etc.).
- Menu para selecionar o modelo de processamento da webcam (OpenCV, MediaPipe, YOLOv8 ou FaceXFormer).
- Seleção entre processamento via CPU ou GPU.
- Opção de desfocar rostos para privacidade.
- Armazenamento de resultados em PostgreSQL (via `POSTGRES_DSN`).
- `PHOTOS_DIR`: pasta onde ficam as fotos capturadas.
- `RF_THRESHOLD`: limiar para comparacao de embeddings (padrao 0.6)
- Interface web em Flask e Dockerfile para facilitar a execução.
- Tradução de fala em tempo real via OpenAI Whisper (use `--webcam` para traduzir enquanto a webcam está aberta).
- A tradução pode ser ativada ou desativada a qualquer momento no menu **Outros**.
- É possível escolher o idioma de entrada e o de saída para tradução.

## Servidor web

Execute o servidor Flask com:

```bash
face-web
```

Ou, se preferir, ainda é possível usar:

```bash
python -m reconhecimento_facial.web_app
```

Ele ficará escutando em `0.0.0.0:8000`. Exemplos de requisições:

```bash
# Processa uma imagem
curl -F 'file=@foto.jpg' http://localhost:8000/process

# Lista pessoas cadastradas
curl http://localhost:8000/people

# Lista as últimas detecções
curl http://localhost:8000/detections
```

Também é possível acessar uma interface gráfica completa via navegador em
`http://localhost:8000/`. A página inicial apresenta um menu organizado em
categorias (Detecção, Reconhecimento, Cadastrar pessoa, Gerenciar pessoas e
Outros) que dá acesso a todas as funcionalidades oferecidas na linha de
comando.

Copie o arquivo `.env.example` para `.env` e ajuste conforme necessário. Todas as dependências podem ser instaladas utilizando o `pyproject.toml`. A variável `POSTGRES_DSN` **deve** ser definida nesse arquivo caso queira usar o banco de dados.

## Variáveis de ambiente

- `MEDIAPIPE_REPO`: repositório do modelo MediaPipe (padrão: `qualcomm/MediaPipe-Face-Detection`).
- `YOLOV8_REPO`: repositório do modelo YOLOv8 (padrão: `jaredthejelly/yolov8s-face-detection`).
- `HF_CAPTION_MODEL`: modelo de legenda (padrão: `nlpconnect/vit-gpt2-image-captioning`).
- `OBSTRUCTION_MODEL_REPO`: modelo para detectar obstrução facial (padrão: `dima806/face_obstruction_image_detection`).
- `FACEXFORMER_REPO`: repositório do FaceXFormer utilizado nas funções de demografia e `analyze_face`.
- `WHISPER_MODEL`: modelo padrão do Whisper para tradução de áudio (padrão: `openai/whisper-large-v3-turbo`).
- `RF_DEVICE`: define o dispositivo de processamento (`auto`, `cpu` ou `gpu`).
- `POSTGRES_DSN`: string de conexão do PostgreSQL usada por `db.py` (sem valor padrão).
- `PHOTOS_DIR`: pasta onde ficam as fotos capturadas.

### Graylog e permissoes de escrita
Caso utilize um container que tambem execute o Graylog, defina a variavel `GRAYLOG_NODE_ID_FILE` para um caminho gravavel, como `/usr/share/graylog/data/config/node-id`. Se o Graylog tentar usar `/etc/graylog/server` sem permissao, o container reiniciara continuamente.


## Requisitos

- Python 3
- opencv-python
- mediapipe
- ultralytics
- torch >=2.6
- transformers
- huggingface_hub
- python-dotenv
- sounddevice

## Testes

Instale as dependências mínimas de testes e execute os testes automatizados com:

```bash
pip install -r requirements-test.txt
pytest
```

## Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para detalhes.


