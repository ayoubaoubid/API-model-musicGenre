from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import shutil
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import librosa

def extract_features(file):
    try:
        audio, sr = librosa.load(file, duration=30)
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=128  
        )  # shape = (128, T) où T dépend de la durée
        
        # 🔹 Ajuster à 128 frames (padding ou découpage)
        if mfcc.shape[1] < 128:
            pad_width = 128 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :128]
        
        # 🔹 Reshape en (128,128,1) pour Conv2D
        mfcc = mfcc.reshape(128,128,1)
        
        return mfcc
    
    except Exception as e:
        print("Erreur sur fichier :", file, e)
        return None
    
    
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