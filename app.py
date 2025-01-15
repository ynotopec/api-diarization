import requests
import numpy as np
import datetime
import threading
import uuid
import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Paramètres de l'API
API_URL = "https://api-audio2embedding.c0.cloud-pi-native.com/get_embedding/"

# Chemin de stockage des embeddings
EMBEDDING_STORE = "embeddings.json"

# Initialiser l'application FastAPI
app = FastAPI()

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permettre à toutes les origines (à restreindre en production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction pour obtenir un embedding à partir d'un fichier audio
def get_embedding(file_path):
    try:
        with open(file_path, 'rb') as file:
            response = requests.post(
                API_URL,
                headers={
                    "accept": "application/json"
                },
                files={"file": (os.path.basename(file_path), file, "audio/opus")}
            )
            # Ajouter un log de la réponse pour le débogage
            print(f"Statut de la réponse de l'API : {response.status_code}")
            print(f"Contenu de la réponse : {response.text}")
            
            response.raise_for_status()  # Lève une erreur si le statut HTTP est une erreur
            return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erreur lors de l'appel à l'API externe : {e}")

# Charger les embeddings existants
def load_embeddings():
    if not os.path.exists(EMBEDDING_STORE):
        return {}
    with open(EMBEDDING_STORE, 'r') as f:
        return json.load(f)

# Sauvegarder les embeddings
def save_embeddings(embeddings):
    with open(EMBEDDING_STORE, 'w') as f:
        json.dump(embeddings, f, indent=4)

# Calculer la similarité entre deux embeddings (cosine similarity)
def cosine_similarity(embedding1, embedding2):
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Définir une fonction pour moyenner deux embeddings
def average_embeddings(embedding1, embedding2):
    return ((np.array(embedding1) + np.array(embedding2)) / 2).tolist()
#    return ((2 * np.array(embedding1) + np.array(embedding2)) / 3).tolist()

# Ajouter ou mettre à jour un embedding dans le stockage
def add_or_update_embedding(new_embedding, similarity_threshold=0.6):
    embeddings = load_embeddings()
    today_str = datetime.datetime.now().isoformat()
    
    # Vérifier la similarité avec les embeddings existants
    for identifier, data in embeddings.items():
        stored_embedding = data["embedding"]
        similarity = cosine_similarity(new_embedding, stored_embedding)
        if similarity >= similarity_threshold:
            # Mettre à jour l'embedding en le moyennant avec le nouveau
            embeddings[identifier]["embedding"] = average_embeddings(stored_embedding, new_embedding)
            embeddings[identifier]["date"] = today_str
            save_embeddings(embeddings)
            return identifier

    # Sinon, ajouter un nouveau
    new_identifier = str(uuid.uuid4())
    embeddings[new_identifier] = {
        "embedding": new_embedding,
        "date": today_str
    }
    save_embeddings(embeddings)
    return new_identifier

# Supprimer les embeddings plus anciens qu'une certaine date
def delete_old_embeddings():
    embeddings = load_embeddings()
    cutoff_date = datetime.datetime.now() - datetime.timedelta(hours=1)
    cutoff_date_str = cutoff_date.isoformat()
    updated_embeddings = {k: v for k, v in embeddings.items() if v["date"] > cutoff_date_str}
    save_embeddings(updated_embeddings)

# Supprimer automatiquement les embeddings plus anciens qu'une heure

def start_automatic_deletion():
    def delete_periodically():
        while True:
            delete_old_embeddings()
            threading.Event().wait(3600)  # Attendre une heure avant la prochaine suppression

    thread = threading.Thread(target=delete_periodically, daemon=True)
    thread.start()

# Démarrer la suppression automatique
start_automatic_deletion()

# Endpoint FastAPI pour uploader un fichier audio et obtenir l'identifiant unique
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    # Sauvegarder temporairement le fichier uploadé
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Obtenir l'embedding du fichier audio
        try:
            new_embedding = get_embedding(temp_file_path)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        # Ajouter ou mettre à jour l'embedding et obtenir l'identifiant unique
        identifier = add_or_update_embedding(new_embedding)
        
        return {"identifier": identifier}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier : {e}")
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



async def upload_audio(file: UploadFile = File(...)):
    # Sauvegarder temporairement le fichier uploadé
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Obtenir l'embedding du fichier audio
        try:
            new_embedding = get_embedding(temp_file_path)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        # Ajouter ou mettre à jour l'embedding et obtenir l'identifiant unique
        identifier = add_or_update_embedding(new_embedding)
        
        return {"identifier": identifier}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier : {e}")
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Exemple d'utilisation
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
