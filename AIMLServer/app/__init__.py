from flask import Flask
from app.routes.predict import predict_blueprint
from app.routes.train import train_blueprint

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(predict_blueprint, url_prefix='/predict')
app.register_blueprint(train_blueprint, url_prefix='/train')
