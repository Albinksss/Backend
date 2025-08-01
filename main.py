from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import uuid
import os
import pandas as pd
import time
from typing import Optional
from Opti_reseau_Solveur import Opti_reseau_Solveur  # Ton module d'optimisation

app = FastAPI()

# 📁 Répertoire temporaire pour les fichiers
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📦 Cache mémoire pour éviter la relecture
cached_files = {}
file_timestamps = {}
CACHE_TTL_SECONDS = 3600  # 1 heure

# 🔁 Nettoyage du cache expiré
def cleanup_old_cache():
    now = time.time()
    expired = [fid for fid, t in file_timestamps.items() if now - t > CACHE_TTL_SECONDS]
    for fid in expired:
        cached_files.pop(fid, None)
        file_timestamps.pop(fid, None)
        print(f"🧹 Fichier {fid} supprimé du cache (expiré)")

# 📤 Upload d'un fichier Excel
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    cleanup_old_cache()
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.xlsx")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_excel(file_path)
    cached_files[file_id] = df
    file_timestamps[file_id] = time.time()

    return {"file_id": file_id}

# 🚀 Traitement du fichier
@app.post("/process/")
async def process_file(
    file_id: str = Form(...),
    file_id2: str = Form(...),
    optimization: str = Form(...),
    maille: str = Form(...),
    Nb_entrepot: str = Form(...),
    Col_NUTS: str = Form(...),
    param1: Optional[str] = Form(None),
    param2: Optional[str] = Form(None),
    param3: Optional[str] = Form(None),
    poids1: Optional[str] = Form(None),
    poids2: Optional[str] = Form(None),
    poids3: Optional[str] = Form(None),
):
    cleanup_old_cache()

    # 🗂️ Chargement des fichiers depuis le cache (ou disque)
    for fid in [file_id, file_id2]:
        if fid not in cached_files:
            file_path = os.path.join(UPLOAD_FOLDER, f"{fid}.xlsx")
            cached_files[fid] = pd.read_excel(file_path)
            file_timestamps[fid] = time.time()

    df = cached_files[file_id]
    df2 = cached_files[file_id2]

    # ⚖️ Pondérations
    params = []
    if param1 and poids1 is not None:
        params.append((param1, int(poids1)))
    if param2 and poids2 is not None:
        params.append((param2, int(poids2)))
    if param3 and poids3 is not None:
        params.append((param3, int(poids3)))

    print("🧠 Param Solveur:", {
    "nb": Nb_entrepot,
    "maille": maille,
    "Col_NUTS": Col_NUTS,
    "params": params,
    "df.shape": df.shape,
    "df2.shape": df2.shape
    })
    
    # 🚀 Lancement de l'optimisation
    if optimization == "Opti_Solveur":
        df_opt, df_affectation = Opti_reseau_Solveur(
            df, df2, int(Nb_entrepot), maille, Col_NUTS, params
        )

    # if optimization == "Entrepots multiples":
    #     df_opt, df_affectation = Opti_reseau(...)

    # 🌍 Conversion GeoJSON
    df_affectation["geometry"] = df_affectation["geometry"].apply(lambda g: g.__geo_interface__)

    return {
        "entrepots": df_opt.to_dict(orient="records"),
        "affectation": df_affectation.to_dict(orient="records")
    }
