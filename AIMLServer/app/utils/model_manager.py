import joblib

MODEL_PATH = 'app/model/model.pkl'

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise Exception("Model file not found. Train the model first.")

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def make_prediction(model, data):
    return model.predict([data])
