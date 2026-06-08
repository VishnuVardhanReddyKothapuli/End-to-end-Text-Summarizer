FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt setup.py README.md ./
COPY src ./src
COPY config ./config
COPY params.yaml app.py main.py ./

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
