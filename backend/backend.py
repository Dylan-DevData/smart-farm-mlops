from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
import cv2

models_pool = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- DÉBUT DU CHARGEMENT DU MODÈLE KERAS ---")
    try:
        models_pool["corn_model"] = tf.keras.models.load_model("best_model_corn_transfer.keras")
        print("--- MODÈLE KERAS CHARGÉ AVEC SUCCÈS ! SERVEUR PRÊT ---")
    except Exception as e:
        print(f"Erreur critique au chargement du modèle : {e}")
    yield
    models_pool.clear()


app = FastAPI(lifespan=lifespan, title="Smart Farm Inference API")


@app.get("/")
def read_root():
    return {"status": "online", "message": "L'API MLOps fonctionne !"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if "corn_model" not in models_pool:
        return JSONResponse(status_code=503, content={"error": "Modèle non disponible"})

    try:
        # 1. Lecture des bytes de l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Prétraitement (ici format 224x224, à adapter selon ton modèle)
        img_resized = cv2.resize(img, (224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)  # Format (1, 224, 224, 3)

        # 3. Inférence
        predictions = models_pool["corn_model"].predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return {
            "prediction": int(np.argmax(score)),
            "confidence": float(np.max(score))
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur traitement: {str(e)}"})