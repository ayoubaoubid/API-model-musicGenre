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