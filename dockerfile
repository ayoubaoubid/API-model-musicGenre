# image de base
FROM python:3.10

# dossier de travail
WORKDIR /app

# Mettre à jour pip en premier
#RUN pip install --default-timeout=1000 --retries=10 --upgrade pip
RUN pip install --upgrade pip

# 1. Copier UNIQUEMENT le requirements.txt d'abord (pour profiter du cache Docker)
COPY requirements.txt .

# 2. Installer dépendances avec un timeout rallongé
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# 3. Copier le reste des fichiers du projet APRES l'installation
COPY . .

# exposer port
EXPOSE 8000

# lancer API (ATTENTION: host 0.0.0.0 est obligatoire dans Docker)
CMD ["uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"]