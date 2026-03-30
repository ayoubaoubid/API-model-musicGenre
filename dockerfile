# Image de base légère avec Python
FROM python:3.10-slim

# Variables d’environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer dépendances système (IMPORTANT pour librosa & soundfile)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"]