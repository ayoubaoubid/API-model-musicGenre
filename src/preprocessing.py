import os
import librosa
import numpy as np
import pandas as pd

# 🔹 2. Extraction des MFCC (128 coefficients)
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



