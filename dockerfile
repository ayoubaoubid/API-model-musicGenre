# Image de base légère
FROM python:3.10-slim

# Éviter fichiers inutiles
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer dépendances système (IMPORTANT pour librosa)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Créer dossier app
WORKDIR /app

# Copier requirements d'abord (optimisation cache)
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]