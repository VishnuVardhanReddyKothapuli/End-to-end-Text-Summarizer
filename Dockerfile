FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt setup.py README.md ./
COPY src ./src
COPY config ./config
# Create empty artifacts dir (actual artifacts are gitignored; app downloads them at runtime)
RUN mkdir -p ./artifacts
COPY params.yaml app.py main.py ./

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]