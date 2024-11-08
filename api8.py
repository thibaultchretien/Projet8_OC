import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Obtenir les détails des tensors d'entrée et de sortie
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Route home pour vérifier que l'API fonctionne
@app.route('/')
def home():
    return jsonify({'message': 'API de segmentation d\'image en cours de fonctionnement !'}), 200

# Fonction pour prétraiter l'image avant de l'envoyer au modèle
def preprocess_image(image):
    # Redimensionner l'image à la taille d'entrée du modèle
    image = image.resize((256, 256))  # Ajustez la taille si nécessaire
    image = np.array(image) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Ajouter la dimension du batch
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
    
    # Faire une prédiction avec le modèle TensorFlow Lite
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Convertir le masque en image et le renvoyer
    mask = prediction[0]  # Si le modèle renvoie un batch, prendre le premier élément
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binariser le masque si nécessaire
    mask_image = Image.fromarray(mask)

    # Sauvegarder l'image de prédiction et l'envoyer au client
    output_image_path = "predicted_mask.png"
    mask_image.save(output_image_path)
    
    return jsonify({'message': 'Prédiction effectuée avec succès', 'output_image': output_image_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
