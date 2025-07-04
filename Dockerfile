FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pip && pip install --no-cache-dir .
CMD ["python3", "-m", "reconhecimento_facial.app"]
