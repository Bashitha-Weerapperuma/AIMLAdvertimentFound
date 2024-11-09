from flask import Blueprint, jsonify
from app.utils.model_manager import save_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train_blueprint = Blueprint('train', __name__)

@train_blueprint.route('/', methods=['POST'])
def train():
    try:
        # Load and preprocess data
        df = pd.read_csv('data/dataset.csv')
        X = df.drop(columns=['label'])
        y = df['label']

        # Train the model
        model = RandomForestClassifier()
        model.fit(X, y)

        # Save the model
        save_model(model)
        return jsonify({'message': 'Model trained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
