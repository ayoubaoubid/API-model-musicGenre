from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import shutil
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import librosa
import numpy as np

from preprocessing import extract_features

app = FastAPI()

# Charger le modèle une seule fois au démarrage
model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "..", "model", "music_modele.keras")
)

classes = ["andalusian", "chaabi", "gnawa", "imazighn", "rai", "rap"]


@app.get("/")
def home():
    return {"message": "Music Genre Classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) :
    
    # Sauvegarde temporaire
    temp_file = f"temp_{file.filename}"
    
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extraction des features
        features = extract_features(temp_file)

        if features is None:
            return {"error": "Erreur lors du traitement audio"}

        features = features.reshape(-1, 128, 1)
        features = np.expand_dims(features, axis=0) 
        
        # reshape pour le modèle → (1, 13)
        #features = np.array(features)
        #features = features[np.newaxis, :]

        # Prédiction
        prediction = model.predict(features)
        predicted_class = classes[np.argmax(prediction)]
        

        return predicted_class

    except Exception as e:
        return {"error": str(e)}

    finally:
        # supprimer fichier temporaire
        if os.path.exists(temp_file):
            os.remove(temp_file)