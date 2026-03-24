from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import shutil
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import librosa
import numpy as np

def extract_features(file_path):
    """
    Extraire les features MFCC (13,) à partir d’un fichier audio
    ⚠️ IDENTIQUE au training (très important)
    """
    try:
        # charger audio (30 secondes max)
        audio, sr = librosa.load(file_path, duration=30)

        # extraire MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # moyenne sur le temps → shape (13,)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        print(f"Erreur lors du traitement de {file_path} : {e}")
        return None
    
    
#if __name__ == "__main__":
#    # Test rapide
#    features = extract_features("test_audio.wav")

app = FastAPI()

# Charger le modèle une seule fois au démarrage
model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "..", "model", "model_genre_music.h5")
)

classes = ["andalusian", "chaabi", "gnawa", "imazighn", "rai", "rap"]


@app.get("/")
def home():
    return {"message": "Music Genre Classification API is running"}


@app.post("/predict")

async def predict(file: UploadFile = File(...)):
    
    # Sauvegarde temporaire
    temp_file = f"temp_{file.filename}"
    
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extraction des features
        features = extract_features(temp_file)

        if features is None:
            return {"error": "Erreur lors du traitement audio"}

        # reshape pour le modèle → (1, 13)
        features = np.array(features)
        features = features[np.newaxis, :]

        # Prédiction
        prediction = model.predict(features)
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            "prediction": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # supprimer fichier temporaire
        if os.path.exists(temp_file):
            os.remove(temp_file)