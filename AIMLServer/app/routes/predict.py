from flask import Blueprint, request, jsonify
from app.utils.model_manager import load_model, make_prediction
from app.utils.data_processing import preprocess_input

predict_blueprint = Blueprint('predict', __name__)

@predict_blueprint.route('/', methods=['POST'])
def predict():
    try:
        data = request.json
        processed_data = preprocess_input(data)
        model = load_model()
        prediction = make_prediction(model, processed_data)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
