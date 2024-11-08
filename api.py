@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("Erreur : Pas d'image dans la requête")
        return jsonify({'error': 'No image file in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        print("Erreur : Aucune image sélectionnée")
        return jsonify({'error': 'No selected file'}), 400

    # Sauvegarder l'image reçue
    file_path = os.path.join('uploads', file.filename)
    try:
        print(f"Enregistrement de l'image dans {file_path}")
        file.save(file_path)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image : {str(e)}")
        return jsonify({'error': f"Erreur lors de la sauvegarde de l'image: {str(e)}"}), 500

    # Prédire le mask
    mask, error = predict_mask(file_path)
    if mask is None:
        print(f"Erreur lors de la prédiction : {error}")
        return jsonify({'error': f"Erreur lors de la prédiction: {error}"}), 500

    # Sauvegarder le mask prédit
    try:
        mask_image = Image.fromarray(mask.astype(np.uint8))
        mask_image_path = 'predicted_mask.png'
        mask_image.save(mask_image_path)
        print(f"Mask sauvegardé dans {mask_image_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du mask : {str(e)}")
        return jsonify({'error': f"Erreur lors de la sauvegarde du mask: {str(e)}"}), 500

    return jsonify({'message': 'Prediction complete', 'mask_image': mask_image_path})
