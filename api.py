from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import io
import base64

app = Flask(__name__)

model = load_model('model_segmentation.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image = image.resize((256, 256))  # Redimensionner à la taille requise
    image_array = np.array(image) / 255.0  # Normalisation

    # Ajouter la dimension batch pour la prédiction
    image_array = np.expand_dims(image_array, axis=0)

    # Prédire le masque
    mask = model.predict(image_array)

    # Appliquer np.argmax si nécessaire (si multi-classe)
    mask = np.argmax(mask, axis=-1)
    mask = np.squeeze(mask, axis=0)  # Enlever la dimension batch

    # Convertir le masque en image et l'encoder en base64
    mask_image = Image.fromarray(mask.astype(np.uint8))
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({"mask_image_base64": mask_base64})

if __name__ == '__main__':
    app.run(debug=True)
