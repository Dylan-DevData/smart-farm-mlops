import streamlit as st
import requests
from PIL import Image
import io

# URL de ton API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Smart Farm Upload", layout="centered")
st.title("🌽 Analyse d'Images de Cultures (Maïs)")

st.write("Uploadez une photo pour analyser l'état de la culture.")

# Widget de téléchargement d'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Afficher l'image sélectionnée par l'utilisateur
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_container_width=True)

    # Bouton pour déclencher l'analyse
    if st.button("Lancer l'analyse MLOps"):
        # Convertir l'image chargée en bytes pour l'envoi HTTP
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_bytes = img_byte_arr.getvalue()

        with st.spinner("Analyse en cours par l'API..."):
            try:
                # 2. Envoi de l'image à FastAPI
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                )

                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")

                    # 3. Affichage propre des résultats
                    st.success("Analyse terminée avec succès !")
                    st.metric(label="Classe Prédite", value=f"Classe {prediction}")
                    st.metric(label="Score de Confiance", value=f"{confidence * 100:.2f} %")
                else:
                    st.error(f"L'API a renvoyé une erreur : {response.status_code}")

            except requests.exceptions.RequestException:
                st.error("Impossible de contacter l'API Backend. Vérifiez qu'elle est bien allumée.")