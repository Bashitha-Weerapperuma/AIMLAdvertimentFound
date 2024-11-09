import joblib
import os

def save_model(model):
    try:
        # Define the model save path
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
        
        # Save the model using joblib (recommended for large models)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
        return joblib.load(model_path)
    except FileNotFoundError:
        raise Exception("Model file not found. Train the model first.")
    
def make_prediction(model, data):
    return model.predict([data])
