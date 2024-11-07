import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle de segmentation
model = load_model('segmentation_model.h5')

# Route home pour vérifier que l'API fonctionne
@app.route('/')
def home():
    return jsonify({'message': 'API de segmentation d\'image en cours de fonctionnement !'}), 200

# Fonction pour prétraiter l'image avant de l'envoyer au modèle
def preprocess_image(image):
    # Redimensionner l'image à la taille d'entrée du modèle
    image = image.resize((256, 256))  # Ajuste la taille en fonction de ton modèle
    image = np.array(image) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter la dimension du batch
    return image

# Route pour effectuer la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Lire et prétraiter l'image
    image = Image.open(io.BytesIO(file.read()))
    preprocessed_image = preprocess_image(image)
    
    # Faire une prédiction avec le modèle
    prediction = model.predict(preprocessed_image)
    
    # Convertir le masque en image et le renvoyer
    mask = prediction[0]  # Si ton modèle renvoie un batch, prendre le premier élément
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binariser le masque si nécessaire
    mask_image = Image.fromarray(mask)

    # Sauvegarder l'image de prédiction et l'envoyer au client
    output_image_path = "predicted_mask.png"
    mask_image.save(output_image_path)
    
    return jsonify({'message': 'Prédiction effectuée avec succès', 'output_image': output_image_path})

if __name__ == '__main__':
    app.run(debug=True, port=5050)
