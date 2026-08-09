# Deploiement sur un Hugging Face Space, SDK "Docker"
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Valeurs par defaut, a écraser via les "Secrets" du Space (Settings > Variables and secrets)
ENV MODEL_NAME=facebook/nllb-200-distilled-600M
ENV ADMIN_TOKEN=change-me

# Les Spaces Hugging Face exposent le port 7860 par convention
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
